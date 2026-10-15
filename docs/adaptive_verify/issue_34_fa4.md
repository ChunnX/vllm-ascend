# Issue #34：FA4 混合缓存组隔离修复

## 原因与修改

vLLM 0.30 的混合缓存组共享底层分配，通过物理 block ID 隔离。
此前 FA4 将整个 layer buffer 重排为连续 K 区域和 V 区域，破坏了这个对应关系：
draft 写逻辑 block 1 时，可能修改 target 所属的物理 block 2。
上一轮测试只验证了连续性、共享 storage 和逻辑容量，遗漏了跨缓存组隔离。

本次恢复原有分页写入布局，在 FA4 读取端使用连续别名视图及页号映射。
例如物理页为 `[K, V]` 时，FA4 视图中的 block 号变成原 block 号的两倍；
K、V 各自保留原始起始地址。padding 对应的间隙页不被页表引用。
缓存数据不搬运，不额外分配整份 K/V，显存池和 scheduler block ID 保持原有对应关系。
页表乘法在设备执行，可随 ACLGraph 捕获和回放；不读取设备上的长度或页号到 CPU。

当前支持页内连续、K/V 形状相同，且物理页跨度为单份 K 页大小整数倍的 NHD 缓存。
其他布局会明确报错并打印 shape/stride，避免静默错误。

## 已完成的本地验证

CPU 回归覆盖 BF16/FP16、128-token 页、带 padding、非顺序 block ID、最后一个 block、
缓存重复使用及含 NaN 的未使用区域。验证 FA4 别名读取结果与原分页读取一致，
draft 写入不会改变另一个 block 的字节。另有 forward 测试检查实际传给 FA4 的页表和别名。

```bash
python -m pytest --noconftest -q \
  tests/ut/attention/test_fa4_kv_cache_layout.py
```

这些 CPU 测试不等价于真实 FA4 算子和 NPU 图回放验证。

## 服务器验证

1. 停止旧服务，切到 `030_dspark_dev` 分支。
2. 使用原脚本重启，保留 `VLLM_ASCEND_ENABLE_DSPARK_FIA_SINK=2`、图模式、
   async scheduling 和 prefix caching。无需修改启动参数或重新编译 C++ 算子。
   必须重启进程，使缓存视图和 ACLGraph 重新创建。
3. 同一提示词串行请求至少 10 次，再交替使用两个不同提示词，并测试少量并发请求。
   记录是否仍有重复 `!`、接受率和 prefix cache 命中情况。
4. 若仍有问题，反馈当前 commit、异常请求及服务日志。
   若出现布局检查报错，提供报错中的 K/V shape 和 stride。

用户已验证连续请求没有再出现乱码，接受率正常。修复已合入 `030_dspark_dev`，
验证分支已删除，[issue #34](https://github.com/singzhou/vllm-ascend/issues/34) 已关闭。
此记录是当次验证结果；后续源码变化仍需按范围重新验证。
