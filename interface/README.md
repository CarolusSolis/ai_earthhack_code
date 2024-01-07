# SustainaBytes

## Run

### Clone repository
```bash
git clone https://github.com/CarolusSolis/ai_earthhack_code.git
cd ai_earthhack_code/interface
```

### Prep Env & Install dependencies

```bash
conda create -n sustainabytes python=3.9
conda activate sustainabytes
pip install -r requirements.txt
```

### Replace the API key in the .env file

Create a file named `.env` and add the `OPENAI_API_KEY` to it.

### Run the app

```bash
sudo hupper -m streamlit run app.py
```
