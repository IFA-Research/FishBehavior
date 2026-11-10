# Fish Behavior Classification

Classify feeding intensity (Strong, Medium, Weak, None) from a short video clip. The default model is MobileViT-Small followed by a two-layer BiLSTM.

```bash
pip install -r requirements.txt
python main.py --config configs/base.json
```

Results are written to `runs/<experiment_name>/`.

Put the recordings in `./Dataset/Resized/` like this:

```text
Dataset/Resized/<date>/<time>/Labeled_WaterParameter_new.csv
Dataset/Resized/<date>/<time>/left/frame_000001.jpg
```

The label column is `Hunger Level`. To change the temporal block, edit `temporal_type` in the config: `bilstm`, `transformer`, or `none`.
