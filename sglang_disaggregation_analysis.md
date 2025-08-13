# SGLang Disaggregation 模块深度分析

## 一、概述

SGLang 的 disaggregation 模块实现了 **Prefill-Decode (PD) 分离**架构，这是一种将推理过程中的预填充（Prefill）和解码（Decode）阶段分离到不同的服务器上执行的技术。这种设计可以：

1. **提高资源利用率**：预填充和解码有不同的计算特性，分离后可以针对性优化
2. **增强可扩展性**：可以独立扩展预填充和解码服务器
3. **优化延迟**：通过专门的硬件和调度策略优化各自的性能

## 二、核心架构设计

### 2.1 整体架构

```
┌─────────────────┐
│   Load Balancer │ (mini_lb.py / launch_lb.py)
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼────┐
│Prefill│ │Decode │
│Server │ │Server │
└───┬───┘ └───┬───┘
    │         │
    └────┬────┘
         │
    KV Transfer
   (RDMA/TCP/etc)
```

### 2.2 关键组件

1. **Prefill Server** (`prefill.py`)：处理输入序列的预填充
2. **Decode Server** (`decode.py`)：处理自回归解码
3. **Load Balancer** (`mini_lb.py`)：请求路由和负载均衡
4. **KV Transfer** (`kv_events.py`, `conn.py`)：KV缓存传输机制
5. **Utils** (`utils.py`)：共享工具和数据结构

## 三、核心模块详解

### 3.1 Prefill Server (prefill.py)

#### 生命周期管理

Prefill服务器管理请求的三个队列：

```python
# 1. Bootstrap Queue - 初始化和握手
class PrefillBootstrapQueue:
    """存储正在进行初始化的请求"""
    - 初始化每个请求的sender
    - 存储还未完成bootstrap的请求
    - 轮询sender检查bootstrap状态
    - 完成后移至Waiting Queue

# 2. Waiting Queue - 等待处理
    - 使用PrefillAdder弹出请求
    - 运行forward计算
    - 添加到Inflight Queue

# 3. Inflight Queue - 传输中
    - 非阻塞轮询请求的sender
    - 传输完成后返回请求
```

#### 关键类：PrefillBootstrapQueue

```python
class PrefillBootstrapQueue:
    def __init__(self, ...):
        self.token_to_kv_pool = token_to_kv_pool  # KV缓存池
        self.queue: List[Req] = []                # 请求队列
        self.kv_manager = self._init_kv_manager() # KV管理器
        
    def add(self, req: Req):
        """添加请求到bootstrap队列"""
        # 创建KV sender用于传输
        req.disagg_kv_sender = self._create_kv_sender(req)
        
    def pop_bootstrapped(self) -> List[Req]:
        """弹出已完成bootstrap的请求"""
        # 轮询所有sender状态
        # 返回已就绪的请求
```

### 3.2 Decode Server (decode.py)

#### 生命周期管理

Decode服务器的请求处理流程：

```python
# 1. PreallocQueue - 预分配
class DecodePreallocQueue:
    """预分配KV缓存"""
    - 初始化每个请求的receiver
    - 握手并预分配可用的KV缓存
    - 移至TransferQueue

# 2. TransferQueue - 传输
class DecodeTransferQueue:
    """轮询接收KV缓存"""
    - 轮询receiver检查传输状态
    - 传输完成后移至waiting queue

# 3. WaitingQueue - 等待
    - 构建PrebuiltExtendBatch
    - 跳过prefill forward，仅填充元数据

# 4. RunningBatch - 运行
    - 将PrebuiltExtendBatch合并到运行批次
    - 执行解码
```

#### 关键类：DecodeReqToTokenPool

```python
class DecodeReqToTokenPool:
    """
    与ReqToTokenPool的区别：
    - 为预分配请求订阅内存
    - 更灵活的内存管理策略
    """
    def __init__(self, size, max_context_len, device, 
                 enable_memory_saver, pre_alloc_size):
        # size: 最大运行请求数
        # pre_alloc_size: 预分配大小
        self.req_to_token = torch.zeros(
            (size + pre_alloc_size, max_context_len),
            dtype=torch.int32, device=device
        )
```

### 3.3 KV缓存传输机制

#### 3.3.1 事件系统 (kv_events.py)

```python
class KVCacheEvent:
    """KV缓存相关事件基类"""
    
class BlockStored(KVCacheEvent):
    """块存储事件"""
    block_hashes: list[int]
    parent_block_hash: Optional[int]
    token_ids: list[int]
    
class BlockRemoved(KVCacheEvent):
    """块移除事件"""
    block_hashes: list[int]

class EventPublisher(ABC):
    """事件发布器"""
    def publish(self, events: EventBatch):
        """发布事件，保证至少一次传递"""
```

#### 3.3.2 连接抽象 (base/conn.py)

```python
class BaseKVManager(ABC):
    """管理传输状态的基类"""
    
class BaseKVSender(ABC):
    """发送端接口"""
    def init(self, num_kv_indices, aux_index):
        """通知解码服务器KV索引长度"""
    def send(self, kv_indices):
        """发送KV缓存"""
    def poll(self) -> KVPoll:
        """检查传输状态"""
        
class BaseKVReceiver(ABC):
    """接收端接口"""
    def init(self, kv_indices, aux_index):
        """通知预填充服务器KV索引"""
    def poll(self) -> KVPoll:
        """检查传输状态"""
```

### 3.4 传输后端实现

系统支持多种传输后端：

```python
class TransferBackend(Enum):
    MOONCAKE = "mooncake"  # Mooncake RDMA
    NIXL = "nixl"          # NIXL传输
    ASCEND = "ascend"      # 华为昇腾
    FAKE = "fake"          # 测试用假实现
```

每个后端都实现了相应的Manager、Sender、Receiver和BootstrapServer。

### 3.5 负载均衡器 (mini_lb.py)

```python
class MiniLoadBalancer:
    """最小化HTTP负载均衡器"""
    
    def select_pair(self):
        """选择prefill和decode服务器对"""
        prefill_config = random.choice(self.prefill_configs)
        decode_server = random.choice(self.decode_servers)
        return prefill_config.url, prefill_config.bootstrap_port, decode_server
        
    async def generate(self, request, prefill_server, decode_server):
        """并行发送请求到两个服务器"""
        tasks = [
            session.post(f"{prefill_server}/{endpoint}", json=request),
            session.post(f"{decode_server}/{endpoint}", json=request),
        ]
        # 等待两个响应，prefill应该先结束
        prefill_response, decode_response = await asyncio.gather(*tasks)
```

## 四、关键设计模式

### 4.1 Mixin模式

通过Mixin类扩展Scheduler功能：

```python
class SchedulerDisaggregationPrefillMixin:
    """Prefill服务器的调度器扩展"""
    def event_loop_normal_disagg_prefill(self):
        """正常调度循环"""
        
    def event_loop_overlap_disagg_prefill(self):
        """重叠调度循环"""

class SchedulerDisaggregationDecodeMixin:
    """Decode服务器的调度器扩展"""
    def prepare_for_prebuilt_extend(self):
        """准备预构建的扩展批次"""
```

### 4.2 队列管理模式

每个服务器都采用多队列设计：
- **Bootstrap/Prealloc Queue**：初始化和资源分配
- **Transfer Queue**：数据传输管理
- **Waiting Queue**：等待处理
- **Running/Inflight Queue**：执行中的请求

### 4.3 轮询和同步

```python
def poll_and_all_reduce(pollers, gloo_group):
    """轮询并进行all-reduce同步"""
    polls = [int(poller.poll()) for poller in pollers]
    tensor_to_reduce = torch.tensor(polls, dtype=torch.uint8, device="cpu")
    dist.all_reduce(tensor_to_reduce, op=dist.ReduceOp.MIN, group=gloo_group)
    return tensor_to_reduce.tolist()
```

## 五、元数据管理

### 5.1 MetadataBuffers

```python
class MetadataBuffers:
    """管理传输的元数据缓冲区"""
    def __init__(self, size, hidden_size, dtype):
        # 输出token相关
        self.output_ids = torch.zeros((size, 16), dtype=torch.int32)
        self.output_token_logprobs_val = torch.zeros((size, 16), dtype=torch.float32)
        
        # top logprobs
        self.output_top_logprobs_val = torch.zeros((size, max_top_logprobs_num))
        
        # 隐藏状态（用于推测解码）
        self.output_hidden_states = torch.zeros((size, hidden_size), dtype=dtype)
```

### 5.2 页面管理

```python
def kv_to_page_indices(kv_indices: np.ndarray, page_size: int):
    """将KV索引转换为页面索引"""
    # 页面保证满（除最后一页）
    # page_index = kv_index // page_size
    if page_size == 1:
        return kv_indices
    return kv_indices[::page_size] // page_size
```

## 六、故障处理

系统包含完善的故障处理机制：

```python
def prepare_abort(req: Req, error_message: str, status_code: HTTPStatus):
    """准备中止请求"""
    req.finished_reason = FINISH_ABORT(error_message)
    req.status_code = status_code
    
# 测试故障注入
FAILURE_PROB = float(os.getenv("DISAGGREGATION_TEST_FAILURE_PROB", 0))
```

## 七、性能优化

### 7.1 内存管理
- 预分配策略减少动态分配开销
- 页面化管理提高内存利用率
- 支持memory saver模式

### 7.2 并行处理
- TP（张量并行）和DP（数据并行）支持
- 异步传输和计算重叠
- 批处理优化

### 7.3 传输优化
- 支持RDMA高速传输
- 最小化传输数据量（仅传输必要的KV缓存）
- 智能预分配减少等待时间

## 八、使用示例

### 启动负载均衡器
```bash
python -m sglang.srt.disaggregation.launch_lb \
    --prefill http://prefill1:8001 http://prefill2:8002 \
    --decode http://decode1:9001 http://decode2:9002 \
    --port 8000
```

### 配置服务器
```python
# Prefill服务器配置
server_args.disaggregation_mode = DisaggregationMode.PREFILL

# Decode服务器配置  
server_args.disaggregation_mode = DisaggregationMode.DECODE
```

## 九、总结

SGLang的disaggregation模块是一个精心设计的PD分离系统，通过：

1. **清晰的架构分层**：将预填充和解码分离，各自优化
2. **灵活的传输机制**：支持多种传输后端，适应不同硬件
3. **高效的队列管理**：多级队列设计，优化请求处理流程
4. **完善的故障处理**：包含重试、超时、故障注入等机制
5. **性能优化**：内存预分配、并行处理、传输优化等

这种设计使得系统能够在保持高性能的同时，提供良好的可扩展性和可靠性，特别适合大规模LLM推理服务部署。