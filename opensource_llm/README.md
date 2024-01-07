# SustainaBytes

## Run

### Clone repository
```bash
git clone https://github.com/CarolusSolis/ai_earthhack_code.git
cd ai_earthhack_code/opensource_llm
```

### Prep Env & Install dependencies

```bash
conda create -n sustainabytes python=3.10
conda activate sustainabytes
pip install -r requirements.txt
```

### Run the app

```bash
python inference_huggingface.py <dataset_filename>
python filter_ideas.py <dataset_filename>
```

