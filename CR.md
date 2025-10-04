# Fire Detection Filtering Evaluation

## 🔥 Project Overview
This project focuses on improving fire detection reliability in computer vision models.  
After a detection is made, we apply temporal filtering across consecutive frames to verify if the detection persists over time.  
Several filtering strategies (mean, median, sigmoid, etc.) are tested to evaluate their impact on precision and the reduction of false positives.

---

## ⚙️ Environment Setup

### Option 1: Using `venv`
```bash
python -m venv .venv
source .venv/bin/activate      # On Linux/Mac
# or
.venv\Scripts\activate         # On Windows
pip install -r requirements.txt
```

### Option 2: Using `conda`
```bash
conda create -n fp_filtering_method_eval python=3.10
conda activate fp_filtering_method_eval
pip install -r requirements.txt
```

---

## 📁 Data Installation
The project uses two main data archives (not publicly available):

* Alertapi_export_by_type.zip → Contains alerts grouped by category (wildfire, other, low cloud, etc.).

* Sdis77_pred.zip → Contains predictions from the current model. Note: the model changed during August, which may cause mismatches with the sequences above.

Access to these ZIP files must be requested from a team member on Slack (they are not included in the repository).

You can also:

* Use `Run_prediction.py` to recompute predictions if needed.

* Use `New_frame_combine.ipynb` to test filtering methods ("median","mean","max","sigmoid","tanh","relu","leaky_relu","elu","softplus","softmax").

## 📊 Results Report
After testing different filtering methods (mean, median, sigmoid, etc.), the results are largely similar across strategies.  
For this dataset, the **mean remains the most stable method**.

### Critical Notes
- The labeled data is not perfectly accurate, which makes strong conclusions difficult.  
- Given this limitation, it is challenging to assert that any method can consistently outperform the others in detecting false positives.  
- Based on current knowledge and this dataset, **no method demonstrates a clear advantage** in reducing false positives beyond the mean.