import subprocess
import glob
import os


sources = glob.glob("alertapi_export_by_type/**/*")

# --- Methode 1 : utilisation directe du lien Hugging Face ---
weight = "hf://pyronear/yolo11s_mighty-mongoose_v5.1.0/yolo11s_mighty-mongoose_v5.1.0.pt"

# --- Methode 2 : téléchargement local si le fichier n'existe pas ---
# from huggingface_hub import hf_hub_download
# os.makedirs("weights", exist_ok=True)
# local_weight = os.path.join("weights", "yolo11s_mighty-mongoose_v5.1.0.pt")
# if not os.path.exists(local_weight):
#     print("Downloading weights from Hugging Face...")
#     local_weight = hf_hub_download(
#         repo_id="pyronear/yolo11s_mighty-mongoose_v5.1.0",
#         filename="yolo11s_mighty-mongoose_v5.1.0.pt",
#         local_dir="weights"
#     )
# else:
#     print("Weights already exist, using local file.")
# weight = local_weight

for source in sources:
    print(source)
    cmd = f"yolo predict model={weight} iou=0.01 conf=0.05 verbose=False imgsz=1024 source={source} save=True save_txt save_conf project=retex/sdis77_pred name={source.split('/')[-2]}_{source.split('/')[-1]}"
    print(f"* Command:\n{cmd}")
    subprocess.call(cmd, shell=True)
