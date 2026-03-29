# NeuroAdapt deployment checklist

# 1. Validate locally
python -m unittest discover -s tests -v
python inference.py --mode heuristic
openenv validate --json

# 2. Run the API locally
python -m server.app --port 7860

# In another shell:
openenv validate --url http://localhost:7860

# 3. Build and run Docker
docker build -t neuroadapt-env .
docker run -p 7860:7860 neuroadapt-env

# In another shell:
openenv validate --url http://localhost:7860
python inference.py --mode heuristic

# 4. Configure Hugging Face Space variables
# API_BASE_URL=https://router.huggingface.co/v1
# MODEL_NAME=<your-model>
# HF_TOKEN=<your-token>

# 5. Push to a Docker Space tagged with `openenv`
# hf auth login
# git clone https://huggingface.co/spaces/TejaswiKarasani/neuroadapt-env hf-space
# Copy this repo into the cloned Space, then:
# cd hf-space
# git add .
# git commit -m "Deploy NeuroAdapt"
# git push

# 6. Validate the live Space
# openenv validate --url https://tejaswikarasani-neuroadapt-env.hf.space
# python inference.py --mode heuristic
