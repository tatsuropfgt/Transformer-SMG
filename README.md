### Environment
- Python 3.10.10
- CUDA 12.0 (11.2~7くらいでも環境作り直せば動くの確認済み)
```bash
pip install -r requirements.txt
```

### data preparation
```bash
git clone https://github.com/music-x-lab/POP909-Dataset.git {data_dir}
python preprocess_pop909.py --indir {data_dir} --outdir {data_dir}
```
- data_dir: "/n/work3/inaba/dataset/POP909"
- outdir の指定も同じで良い
- outdir/muspy_data 以下に preprocess後のデータがjsonファイル(muspy形式)で保存され，outdir/train.txt, outdir/valid.txt, outdir/test.txtも作成される

### 学習と評価
```bash
python main.py --config {config_path}
```
config_path: (例) configs/note_tuple/baseline.yaml

### 評価のみ
```bash
python evaluate.py --config {config_path}
```


