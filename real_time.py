import cv2
import torch
from modelling.model import build_model
from utils.misc import load_config
import os
# def get_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
#     """
#     Returns the latest checkpoint (by time) from the given directory.
#     If there is no checkpoint in this directory, returns None
#     :param ckpt_dir:
#     :return: latest checkpoint file
#     """
#     list_of_files = glob.glob("{}/*.ckpt".format(ckpt_dir))
#     latest_checkpoint = None
#     if list_of_files:
#         latest_checkpoint = max(list_of_files, key=os.path.getctime)
#     return latest_checkpoint

def load_checkpoint(path: str, map_location: str='cpu') -> dict:
    """
    Load model from saved checkpoint.
    :param path: path to checkpoint
    :param use_cuda: using cuda or not
    :return: checkpoint (dict)
    """
    assert os.path.isfile(path), "Checkpoint %s not found" % path
    checkpoint = torch.load(path, map_location=map_location)
    return checkpoint

# Load configuration and model
#cfg = load_config("experiments/configs/TwoStream/phoenix-2014_keypoint.yaml")
#cfg = load_config("experiments/configs/TwoStream/phoenix-2014_keypoint.yaml")

cfg = load_config("experiments/configs/TwoStream/phoenix-2014_s2g.yaml")
cfg['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(cfg)
#checkpoint = torch.load('results/phoenix-2014_video/ckpts/best.ckpt', map_location='cuda')
#checkpoint = load_checkpoint('results/phoenix-2014_video/ckpts/best.ckpt', map_location='cuda')
checkpoint = load_checkpoint('results/phoenix-2014_s2g/best.ckpt', map_location='cuda')
#checkpoint = load_checkpoint('results/phoenix-2014_keypoint/ckpts/best.ckpt', map_location='cuda')

#checkpoint = torch.load('results/phoenix-2014_keypoint/ckpts/best.ckpt', map_location='cuda')
#model.load_state_dict(checkpoint['model_state'])
model.load_state_dict(checkpoint['model_state'], strict=False)
model.eval()

def preprocess(frame):
    # Implement preprocessing steps here
    return processed_frame

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    processed_frame = preprocess(frame)
    
    with torch.no_grad():
        prediction = model(processed_frame)
    
    # Process prediction to get readable output
    result = process_prediction(prediction)
    
    cv2.putText(frame, str(result), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Real-time SLR', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()