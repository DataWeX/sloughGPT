---
id: 20260802_082152_fix-conv2d-patch-ordering-for-n1-and-batchnorm-training-mode
title: fix conv2d patch-ordering for n>1 and batchnorm training-mode gradients; green new test_slonet_cnn suite
status: done
tags: slonet,cnn,autograd
created: 2026-08-02T08:21:52.446590+00:00
---

fix conv2d patch-ordering for n>1 and batchnorm training-mode gradients; green new test_slonet_cnn suite

Committed 4991f93 (eos/tie/rebuild/warmup fixes; 84 test_slonet_generate green) in prior session.

New session: a concurrent editor added test_slonet_cnn.py (356-line CNN suite) plus forward-path fixes in _conv2d/_batchnorm2d while I fixed the two real autograd bugs it exposed:

1. conv2d backward misordered output channels for batch>1: dY_flat was reshaped to (n*oh*ow, oc) from channel-major (n,oc,oh,ow); now g.transpose(0,2,3,1).reshape(n*oh*ow, oc) in both dX and dW paths. The forward (oc, n_patches) reshape to (n, oc, oh, ow) had the same latent bug (masked at n=1) -- concurrent forward fix reshape(oc,n,oh,ow).transpose(1,0,2,3) complements it; verified exact vs manual im2col JVP at n=2.

2. batchnorm2d training-mode backward/JVP treated batch mean/var as constants. Corrected:
   - backward: g_hat/s - sum(g_hat)/(N*s) - x_center*sum(g_hat*x_center)/(N*s^3)
   - forward_grad: t_mean/t_var/t_norm from tangents
   Both verified numerically (~1e-6).

Also fixed test FD helpers: _fd_grad_x casts to float64 (eps=1e-6) and _naive_conv uses np.result_type, removing float32 cancellation noise (residual 0.037 was FD noise, not a source bug; conv backward now 1.2e-6 vs float64 FD).

Verified:
- test_slonet_cnn.py + test_train_pipeline.py: exit 0
- 9-file slonet sweep: exit 0
- multimodal -m slow (59 tests): exit 0
- 11-suite autograd sweep: exit 0
- backward vs float64 FD: x.grad 2.6e-6, w.grad 1.05e-5

Commits:
- 1a26395 fix(slonet): conv2d patch-ordering for n>1 + batchnorm training-mode gradients (slonet.py + test_slonet_cnn.py)
- 892a219 fix(trainer): EMA first-step uses raw loss; seed generate test (train_pipeline.py + test_train_pipeline.py)

Note: EMA fix guards on explicit None (0 was falsy); test seeds np.random(0). pytest.ini addopts -q -m 'not slow' so multimodal needs explicit -m slow. Known flaky unrelated: test_pugqeep_cache_tasks::TestPGQ::test_submit_training (passes in isolation).