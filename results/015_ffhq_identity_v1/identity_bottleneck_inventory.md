# FFHQ identity bottleneck inventory

Original project: `/mnt/mass_disk1/z.cao/workspace/Submitted/LeapFactual/`.
Read-only audit on 2026-09-17; previous results remain unchanged.

## Original architecture and answers

1. **No explicit task/content decomposition.** `notebooks/Experiment_FFHQ.ipynb` code cells define `Generator.generate`, `wrap_unet1d`, `leap` and `progressive_leap`. The same architecture is in `notebooks/FFHQ_quantitative_eval.py:69` and `models/flows/generate_CE_w_ffhq.py`. A generated sample retains its W code, shape N×512. `Generator.generate` repeats it at all 16 synthesis layers (N×16×512); output N×3×1024×1024, RGB approximately [-1,1]. There is no image encoder/inversion in this pipeline.
2. **No smile-specific bottleneck.** CLIP class conditioning of the flow does not factor W into smile and identity components. Generic VAE bottlenecks elsewhere and MorphoMNIST disentanglement experiments are not FFHQ smile/content components.
3. **Old pipeline modifies the entire W512 code.** The conditional flow is a `KarrasUnet1D(seq_len=512, channels=1, num_classes=2)` operating on N×1×512. Lift/land integration changes W jointly.
4. **No factual identity code is explicitly preserved.** Fixed synthesis noise and pretrained weights are shared across all faces, not factual-specific residual representations.
5. **No original identity-preservation metric was found.** `notebooks/FFHQ_quantitative_eval.py` and `_ref.py` use classification validity, SSIM, PSNR and LPIPS; these are not face-identity embeddings.
6. **No original ArcFace checkpoint or face-identity embedding component was found** in the audited FFHQ sources/checkpoints or targeted checkpoint searches. New independent evaluation uses InsightFace buffalo_l ArcFace with its detector and standard five-landmark alignment. This is not presented as LeapFactual reuse.
7. **Existing checkpoints:** generator `/mnt/big_disk/z.cao/StyleGAN_pkl/stylegan3-r-ffhq-1024x1024.pkl` (`G_ema`); flow `/mnt/mass_disk1/z.cao/workspace/Submitted/LeapFactual/notebooks/checkpoints/ffhq_unet1d_argmax_w_20000_final.pt`; original CLIP ViT-B/32 `/tmp/ViT-B-32.pt`. No trained task/content encoder exists. The new SVD basis is derived from frozen original score gradients, not trained as a new generative model.

## Reused experiment

Parent: `/home/xu.zhao/robust_ce_results/ffhq_grace_v1/20260916_full_v1`.
Training256, calibration64 pool /8 factuals, test8 factuals,8 target references; all frozen IDs, W codes, Base CFs, original classifier, sixteen attack models and thirty future models reused with hashes. Base means ten original progressive-leap iterations, not the five additional robust LeapFactual iterations. Current backbone, text features and temperature stay fixed; shifted-model Update adapts only CLIP visual projection768×512. Literal prompts are `smiling face` and `face`, so labels are prompt-relative pseudo-labels, not verified smile annotations.

## Declared fallback

For all256 training W codes only, differentiate original smiling softmax probability with respect to W512. Compute an uncentered SVD of the256×512 gradient matrix. Select the smallest k explaining at least95% squared singular-value energy. U has shape512×k. No final test outcomes determine k.

Task coordinates: `u=h @ U`. Factual content: `v=h_fact - (h_fact @ U) @ U.T`. Realization: `h'=v+u' @ U.T`, then original StyleGAN synthesis. Only k-dimensional u is optimized; each row keeps its original factual v, including when OT permutes target-task anchors. Geometry is exact Euclidean W2 on u. The only geometry budget is beta×D. No identity penalty or current-validity branch is added.

This is an approximate task/content split. It does not establish classification independence or identity preservation. Identical-capacity linear probes use512-dimensional task/content projections, C=1, training256 and held-out calibration64. Reconstruction is checked on all64 calibration W codes. ArcFace similarity and visual changes are evaluated independently after freezing the run.

The original Base W is projected into task coordinates then realized with factual content. This can differ from the original Base image; report `Conditional_Base` as a diagnostic in addition to the requested original Base comparator.

## New identity evaluation

Official source: https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
Documentation: https://github.com/deepinsight/insightface/tree/master/python-package
InsightFace0.7.3, ONNXRuntime1.22.1. `w600k_r50.onnx`: aligned112×112 RGB face to512-dimensional ArcFace embedding; `det_10g.onnx`: face boxes and five landmarks, detector input640×640. Largest detected face chosen deterministically; detection failures and multiple detections recorded, never silently dropped. Embeddings L2-normalized; cosine similarity reported with mean, population std and10th percentile. Pretrained models are upstream non-commercial research assets, staged externally and copied only to the new raw run with SHA256 hashes.

## Comparison and interpretation controls

Select r and beta from calibration only. Full-latent comparator uses exactly the selected r, bank and beta. Reuse its saved output when settings match; otherwise compute only the previously absent full-latent configuration. The future models have been evaluated in the parent experiment: this is a matched follow-up, not a fresh unseen-shift benchmark. Fix all eight test IDs for the qualitative panels before evaluating the new outputs; no visual subset selection.
