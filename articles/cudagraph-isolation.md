# Eight probes, one bug, one workaround

*Isolating a TurboQuant × speculative-decoding CUDA graph corruption in vLLM, the wrong way and the right way.*

> Living draft, 2026-04-25. Will evolve as upstream lands a fix and as more rigs report in. Companion to the [main repo write-up](https://medium.com/) and to [vllm-project/vllm#40831](https://github.com/vllm-project/vllm/issues/40831).

---

## TL;DR

vLLM's TurboQuant KV cache combined with **any** speculative-decoding method produces degenerate token loops on tool calls, long-context recall, and structured outputs. Eight probes across two models and four hardware rigs isolate the bug to the **CUDA graph capture/replay layer specifically** — not the Triton kernels, not torch.compile inductor output, not attention math. Disabling cudagraph (keeping torch.compile on) fully restores correctness at a ~60% TPS cost. Root cause within the cudagraph layer is still TBD upstream; PR #40798 looked structurally promising but a backport+test on our pinned image showed the bug persists despite all of #40798's changes being live.

---

## The problem

The repo's [original write-up](https://medium.com/) reported 85–106 TPS at 125K context for Qwen3.6-27B on a single RTX 3090, using TurboQuant 3-bit KV cache + MTP speculative decoding. Headline numbers were measured on plain narrative and code completion; both came back fine.

After publication, broader functional testing surfaced something nobody had measured before:

- **Tool calls** emit `<tool_call>` as inline text and never populate `tool_calls[]`. The output dissolves into nested `</parameter>` tails until `max_tokens`.
- **Long-context recall** retrieves the first token of the secret then loops: `"amber amber amber amber..."`, `"violet violet otter 2200"`, `"turqurururururururur..."`.
- **Streaming** occasionally produces token duplication: `"Note: The above above above above..."`.

The 125K KV pool was fine. The problem was the *attention output quality* under the very narrow workloads where output structure matters.

The bug is robust: Qwen3.6-27B on 1× RTX 3090 (Ampere SM 8.6), Qwen3-Next-35B-A3B-FP8 on 2× RTX A5000 (per @Sandermage), Qwen3-Next-35B-A3B-AWQ on RTX 5090, MiMo-7B-RL dense on the same single 3090, all reproduce the same failure shape with their respective TurboQuant preset + MTP or ngram spec-decode.

So: not Lorbus-specific, not 27B-specific, not hybrid-attention-specific, not MoE-specific, not draft-method-specific. Something deeper.

---

## The hypothesis I started with (and why I had to throw it away)

When I first read the Triton decode kernel for TurboQuant, one routing decision in `_prefill_attention` jumped out. For continuation chunks (where `q_len < seq_len` — i.e. the current batch attends to previously cached K/V), the kernel selects between two paths:

```python
if q_len <= _CONTINUATION_DECODE_THRESHOLD:   # = 128
    # synthetic-decode fast path: read ALL K/V from TQ cache,
    # including K/V for positions just stored this step
    out = triton_turboquant_decode_attention(query=q_seq, seq_lens=synth_seq_lens, ...)
else:
    # _continuation_prefill: dequant cached K/V, concat with RAW current chunk K/V,
    # then flash_attn(causal=True)
    out = self._continuation_prefill(...)
```

For spec-decode, q_len is the verify batch size — 1 + num_speculative_tokens, typically 4 or smaller. So spec-decode always takes the synthetic-decode fast path. That path reads even the just-stored draft K/V back through the quantized cache, where TurboQuant's MSE quantization (paper-correct but lossy on inner products) compounds across the within-batch positions.

The hypothesis was clean and paper-backed: TurboQuant's MSE quantizer is mathematically known to bias inner-product estimates ([Zandieh et al. 2025](https://arxiv.org/abs/2504.19874) §3.2 — "TurboQuantmse does not provide unbiased inner product estimates with query vectors ... a multiplicative bias of 2/π"). Within-batch attention compounds the bias across draft positions; outside-batch attention doesn't.

The fix should have been routing-only: change the threshold so spec-decode never takes the synthetic-decode fast path, force it through `_continuation_prefill` (which keeps the current-chunk K/V as raw, unquantized, fp16). I wrote a 5-line patch: `_CONTINUATION_DECODE_THRESHOLD = 0`. Spec-decode q_len > 0 always falls through to `_continuation_prefill`. Within-batch attention now uses raw K. Should be done.

It wasn't done. The probe failed. Same loops, same shape.

This was Probe 4. Sander, working independently, arrived at the architecturally-equivalent fix (`q_len == 1` instead of `q_len <= 128`) and called it [P56](https://github.com/Sandermage/genesis-vllm-patches/blob/main/vllm/_genesis/wiring/patch_56_spec_decode_decode_path_guard.py). Same result on his rig — the catastrophic loops shrank but token-level duplication (`for for x in arr if arr if x`, `age age=`, `<parameter=parameter=>`) persisted.

Two of us, two rigs, same routing fix, same partial result. The bug was below the routing layer.

---

## The single probe that broke it open

Here's the trick: when you've ruled out the hypothesis you walked in with, the cheapest way to make progress is to test the most powerful axis you haven't touched yet.

vLLM has two layers of compilation machinery on top of model dispatch:

1. **torch.compile** (Inductor) — generates fused kernels at warmup time.
2. **CUDA graph capture/replay** — captures GPU-side operations into a graph that replays on subsequent calls.

`--enforce-eager` disables **both** in one flag. If the bug disappears, we know it's in compilation machinery. If the bug persists, it's in the kernels or the dispatch logic, and we need different probes.

I ran it. Bug went away. Every test in `verify-full.sh` passed. The cost was painful (23 TPS down from ~85), but it told me the bug was in compilation, not in attention math. **Probe 5.**

That conflated two layers though. To single-axis it, I added one more probe: `--compilation-config '{"cudagraph_mode":"NONE"}'` keeps torch.compile inductor on but disables CUDA graph capture entirely. **Probe 6.**

Bug went away. 33 TPS — better than enforce_eager because Inductor's kernel fusions are still active, just no graph capture.

That's the single-axis cut: **CUDA graph capture is corrupting the output**. torch.compile is fine. The Triton kernels are fine when invoked dynamically. Only the captured graph, replayed under spec-decode's runtime shapes, produces the corruption.

Three things made probe 6 the inflection point:

1. It empirically distinguished kernel-math from compilation-machinery in a single test.
2. It produced a working configuration that I could ship today (cudagraph-off variant in `docker-compose.longctx-experimental.yml`).
3. It told me what to look for in the captured graph — *something* that has different addresses or shapes between warmup-time capture and runtime replay.

Probe 7 — running Sander's exact 9-prompt corruption-detection suite against this cudagraph-off config — confirmed: 9/9 pass. Tool calls clean (`tool_calls=[get_weather, args={"city":"Paris","unit":"Celsius"}]`), code clean (no `for for`), XML clean (no `age age=`), needles clean. Whatever Sander had been calling "Layer 2 token duplication" was the same root cause as the catastrophic Layer 1 loops, just at a different severity — both vanished together when cudagraph capture went away.

---

## The plot twist

Sander identified a structural fix candidate: PR [#40798](https://github.com/vllm-project/vllm/pull/40798) ("Share decode scratch workspace across layers"). It moves `_tq_mid_o_buf` / `_tq_output_buf` / `_tq_lse_buf` from per-layer `register_buffer(B=max_num_seqs=1)` to `WorkspaceManager.get_simultaneous()` — a persistent base buffer with a stable `data_ptr` that runtime spec-decode shapes can slice into.

The mechanism it would close was textbook: at warmup, capture happens with `B=max_num_seqs=1` (the smallest batch). The captured cudagraph references the buffer's data_ptr at that shape. At runtime, spec-decode verify hits the kernel with `B=q_len=4`. The reuse logic reallocates the buffer to a larger shape and replaces `buf_holder._tq_mid_o_buf` with the new tensor — but the captured cudagraph still references the original (now freed or replaced) data_ptr. Stale pointer → corrupted reads at replay → token loops or duplications depending on which addresses happen to be stale.

Probe 8 backported the full PR via a 9-anchor disk-edit script onto our pinned vLLM nightly. All anchors matched cleanly. `docker exec` confirmed the new code was present in the running container — `register_buffer` block replaced by the comment, `_decode_attention` no longer passing buf kwargs, `is_workspace_manager_initialized()` block in `triton_turboquant_decode.py`, `_reserve_turboquant_decode_workspace()` called at the start of `capture_model`. TPS came in at 96 — confirming both cudagraph and torch.compile were genuinely engaged.

The bug persisted. Same shape. Tool calls fail, recall loops, streaming shows `"above above above"`.

That was useful negative data. Either:

- PR #40798 is necessary but not sufficient; some companion change in `main` is also required (we backported only the four files in the PR diff, against a pinned digest from late April).
- Our backport has a subtle anchor mismatch we missed (less likely given runtime TPS confirmation, but possible — a clean main+#40798 CI build would clarify).
- The buffer-pointer-drift mechanism isn't actually the bug. The captured graph is corrupting the output some other way that #40798 doesn't touch.

Whatever it is, the buffer-stability fix alone isn't enough. Posted upstream and retracted the "this is the likely fix" framing.

---

## What we know now

| Layer | Status |
|---|---|
| Triton store kernel | ✅ correct (probes 1, 5, 6) |
| Triton decode kernel | ✅ correct dynamically (probes 1, 5, 6) |
| TurboQuant attention math | ✅ correct (probe 1: turboquant alone passes everything) |
| Spec-decode framework orchestration | ✅ correct (probe 5: works with everything off) |
| torch.compile inductor output | ✅ correct (probe 6: works with compile on, cudagraph off) |
| **CUDA graph capture/replay** | **✗ corrupts spec-decode + TurboQuant output (mechanism unknown)** |
| PR #40798 workspace-manager refactor | ✗ insufficient on its own (probe 8) |

So the root cause is somewhere in *what gets captured* into the cudagraph for the TurboQuant attention path under spec-decode shapes — but it's not the per-layer scratch buffer pointers, or at least not just those. Could be:

- Stride values baked into the captured graph that don't match runtime tensor strides.
- Slot-mapping or block-table addressing that's specialized at warmup and wrong at runtime.
- Some metadata-driven dispatch (TurboQuant has explicit `is_prefill=(max_query_len > 1)` branching) that captures the wrong branch.
- Workspace acquisition timing — workspace manager state captured during one phase doesn't match the state at replay.

These are all guesses. The one piece of empirical signal is that **the bug happens at exactly the layer where Python code stops running** (cudagraph captures GPU-side ops, not Python).

---

## The shipped workaround

`docker-compose.longctx-experimental.yml` in the repo bakes in `--compilation-config '{"cudagraph_mode":"NONE"}'`. Cost: 33 TPS sustained vs ~85 TPS broken-but-fast. All seven `verify-full.sh` tests pass plus all nine of Sander's Layer-2-detection prompts. 125K context, vision, tools, recall, streaming — all functional.

When upstream lands the fix, that flag drops out and TPS recovers.

The default config (`docker-compose.yml`) wasn't affected by any of this — it ships fp8_e5m2 KV at 20K context, ~85 TPS peak, zero workaround needed. fp8 KV doesn't go through TurboQuant's custom attention backend, so the captured-cudagraph corruption never gets a chance to happen.

---

## What didn't work, in order

1. **`_CONTINUATION_DECODE_THRESHOLD = 0`** (Probe 4). Routes spec-decode out of the synthetic-decode fast path. Should have neutralized within-batch quant bias. Bug persisted. Sander's P56 reached the same architectural endpoint via different anchors and observed the same partial-improvement-only result.
2. **PR #40798 backport** (Probe 8). Moves scratch buffers to a persistent workspace with stable data_ptr. Should have neutralized warmup-vs-runtime pointer drift. Bug persisted at full compile+cudagraph speed.

Both were structurally clean hypotheses with paper-backed or maintainer-flagged reasoning. Both turned out to be at the wrong layer. The lesson is the obvious one: **test your hypotheses against the actual bug, not against your model of the bug.**

---

## Open questions

For maintainers reading this and considering a fix:

1. What capture-time vs replay-time state can a Triton kernel's metadata carry that isn't visible from the Python-level tensor pointer? (Strides, slot offsets, internal scratch pointers, etc.)
2. Does `cudagraph_specialize_lora=True` or `cudagraph_copy_inputs=False` interact with TurboQuant's metadata-driven branching in a way that produces specialization mismatches?
3. Is there a way to dump the captured graph for the TurboQuant attention path and compare it to a non-captured execution at the kernel-argument level?
4. PR #40798 looked structurally promising. Are there companion changes in `main` we'd need alongside it for the workspace mechanism to actually stabilize the captured pointers?

If any maintainer wants to verify our negative result for #40798 against a clean main + PR build, the disk-edit patch we used is in [`patches/patch_pr40798_workspace.py`](../patches/patch_pr40798_workspace.py) — kept in the tree as a research artifact, with a header explaining that it does NOT fix the bug.

---

## Methodology notes

A few things that helped, in case anyone else hits a similar isolation problem:

**Single-axis probes.** Probe 5 (`enforce_eager`) was a two-axis cut (compile + cudagraph). Probe 6 (`cudagraph_mode=NONE`) was the single-axis follow-up that actually disambiguated. The rule: when a probe's result is ambiguous about which axis it changed, run a tighter probe before celebrating.

**Test the hypothesis you walked in with first.** I came in with a paper-backed bias-compounding hypothesis. Probe 4 was specifically designed to test that hypothesis on our config. When it failed, I knew I was wrong before committing to a fix. The cheapest probes are the ones that disprove your priors.

**Negative results are publishable.** Probe 8 was a clean negative: PR #40798 didn't fix it. That information is more useful to upstream than another rig confirming the bug exists. It rules out a structural hypothesis cleanly and tells the next person who looks at #40831 not to walk that path.

**Keep the failing config alive.** Throughout the investigation, the failing config was always one `docker compose up -d` away. Each probe was 30 minutes of teardown + container boot + test + restore. If we'd had to rebuild from source for each probe, the eight-probe ladder would have been a week of work instead of an afternoon.

**Document the path, not just the destination.** The repo's README at this point shows the full probe ladder with TPS for each cell, links to the upstream comments where the data was filed, and points at the disk-edit patches for the two hypotheses we tested and rejected. The next person to pick up #40831 doesn't have to redo any of this.

---

## What's next

This article will get updated as upstream progresses. Specifically:

- If #40798 lands and someone validates against #40831's failure prompts, that result goes here.
- If a different PR lands that closes the bug, the working configuration TPS goes here too.
- If Sander runs probe 6 on his 2× A5000 rig and the cudagraph-off recovery reproduces, that's a fourth-rig confirmation of the workaround.
- If the root cause within the cudagraph layer is identified, the "what we know now" table gets a new row.

The eight-probe ladder isn't necessarily complete. There's one more direction worth exploring if upstream stalls: capture the graph, then dump tensor pointers and strides at replay time vs at capture time to see what specifically diverged. But that crosses from Python-level investigation into kernel-level instrumentation, which is upstream-engineer territory.

For now, the practical user-facing message is the same as it was after probe 6: if you need 125K context with TurboQuant on a single Ampere card today, ship `--compilation-config '{"cudagraph_mode":"NONE"}'` and accept the 33 TPS sustained. It works. When upstream lands a fix, drop the flag.

---

## Acknowledgments

@Sandermage walked into a blue/green window on his own rig, ran our probe suite, identified PR #40798, marked his own routing-fix patch as superseded by our cudagraph-off workaround, and engaged honestly with each negative result. The Genesis patches that make TurboQuant work on consumer Ampere at all are his work; this article is more focused because his comments on #40831 already cleared the ground.

@vibhavagarwal5 wrote the original TurboQuant landing PR (#38479) and the tracking issue (#40069) that made the "spec-decode is unverified" status visible upfront. The gap between an unverified-feature-combination and a known-bug is exactly the gap this investigation crossed.

The Lorbus AutoRound INT4 quant of Qwen3.6-27B with preserved BF16 `mtp.fc` is what made any of this possible — without that, MTP loads with zero parameters and the spec-decode path never even runs.

---

*Filed under: vLLM serving on consumer Ampere; investigations that started from a wrong hypothesis and got somewhere useful anyway. Code: [github.com/noonghunna/qwen36-27b-single-3090](https://github.com/noonghunna/qwen36-27b-single-3090). Bugs: [#40807](https://github.com/vllm-project/vllm/issues/40807) and [#40831](https://github.com/vllm-project/vllm/issues/40831).*
