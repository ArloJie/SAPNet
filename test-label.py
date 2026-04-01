import sys
import copy
from SDRD import *
from datasets import *
from utils import *
from PIL import Image
from metric import *
import scipy.ndimage as ndi
import torch.nn.functional as F
import cv2

# Argument Parser Setup
parser = argparse.ArgumentParser()

###############################################################################
# Dataset Arguments
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=4, type=int)  # Set to 4 for parallelism in dataset loading
parser.add_argument('--data_dir', default=r'/path/to/dataset/', type=str)  # Change this to your dataset directory
parser.add_argument('--dataset', default=r'COD10K', type=str)  # Change dataset

###############################################################################
# Inference Parameters
###############################################################################
parser.add_argument('--tag', default='COD10K', type=str)
parser.add_argument('--domain', default='val', type=str)
parser.add_argument('--vis_dir', default='vis_cam', type=str)

parser.add_argument('--scales', default='0.5,1.0,1.5,2.0', type=str)

if __name__ == '__main__':
    ###################################################################################
    # Arguments Parsing
    ###################################################################################
    args = parser.parse_args()

    experiment_name = args.tag

    if 'train' in args.domain:
        experiment_name += '@train'
    else:
        experiment_name += '@val'

    experiment_name += f'@scale={args.scales}'

    # Create directories for saving results
    pred_dir = create_directory(f'./experiments/predictions/{experiment_name}/')
    cam_path = create_directory(f'{args.vis_dir}/{experiment_name}')

    # Model path setup
    model_path = './labels/models/' + f'-.pth'
    print(f'Model Path: {model_path}')

    # Set random seed for reproducibility
    set_seed(args.seed)
    log_func = lambda string='': print(string)

    ###################################################################################
    # Dataset and DataLoader
    ###################################################################################
    dataset = Test_label_Datasets(args.data_dir, args.dataset)

    ###################################################################################
    # Network Setup
    ###################################################################################
    model = SDRDnet()
    model = model.cuda()
    model.eval()

    log_func(f'[i] Total Params: {calculate_parameters(model)/1e6:.2f}M')
    log_func()

    # Load the model weights
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt)

    MR = MetricRecorder(len(dataset))

    ###################################################################################
    # Evaluation
    ###################################################################################
    eval_timer = Timer()
    scales = [float(scale) for scale in args.scales.split(',')]

    model.eval()
    eval_timer.tik()

    flag = False

    def get_cam(token, scale):
        """Preprocess token and perform inference to get CAM."""
        trans_token = copy.deepcopy(token)
        trans_token = trans_token.cpu().numpy().transpose((1, 2, 0))

        # Resize token according to the given scale
        trans_token = cv2.resize(trans_token, (round(40 * scale), round(40 * scale)), interpolation=cv2.INTER_CUBIC)
        trans_token = trans_token.transpose((2, 0, 1))
        trans_token = torch.from_numpy(trans_token)

        # Create flipped version of the token
        flipped_trans_token = trans_token.flip(-1)

        # Stack both versions (original + flipped) and move to GPU
        trans_token = torch.stack([trans_token, flipped_trans_token])
        trans_token = trans_token.cuda()

        # Inference
        _, _, cams = model(trans_token, inference=True)

        # Flip back the CAMs if needed
        if flag:
            cams = 1 - cams

        # Post-processing
        cams = F.relu(cams)
        cams = cams[0] + cams[1].flip(-1)

        return cams

    # Visualization Flag
    vis_cam = True
    with torch.no_grad():
        length = len(dataset)
        for step, (ori_image, token, image_id, tensor_image, gt) in enumerate(dataset):
            ori_w, ori_h = ori_image.size
            images = tensor_image.cuda()
            label = np.array([1])

            # Calculate strided-up size
            strided_up_size = get_strided_up_size((ori_h, ori_w), 16)
            cams_list = [get_cam(token, scale) for scale in scales]

            # Resize and combine CAMs
            hr_cams_list = [resize_for_tensors(cams.unsqueeze(0), strided_up_size)[0] for cams in cams_list]
            hr_cams = torch.sum(torch.stack(hr_cams_list), dim=0)[:, :ori_h, :ori_w]

            keys = torch.nonzero(torch.from_numpy(label))[:, 0]
            hr_cams = hr_cams[keys]
            hr_cams /= F.adaptive_max_pool2d(hr_cams, (1, 1)) + 1e-5  # Normalize CAM

            ##################################################################
            # Combine the CAMs and resize to original image dimensions
            cam = torch.sum(hr_cams, dim=0)  # Combine CAMs
            cam = cam.unsqueeze(0).unsqueeze(0)
            cam = make_cam(cam)  # Make final CAM

            # Convert CAM to numpy and save
            cam = get_numpy_from_tensor(cam.squeeze())
            image = np.array(ori_image)
            h, w, c = image.shape

            image_id = image_id.split('.')[0]
            image_id_cam = f'{image_id}cam'

            # Save the CAM as an image
            cam = (cam * 255).astype(np.uint8)
            cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

            # images = images.unsqueeze(0).clone().detach()
            # for i in range(images.shape[0]):
            #     grid = utils.make_grid(images[i].unsqueeze(0), nrow=1, padding=0, pad_value=0, normalize=True)
            #     imagek = grid.mul_(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()[..., ::-1]
            # cv2.imwrite(f'{cam_path}/{image_id}.jpg', imagek)

            # Save the visualization
            cv2.imwrite(f'{cam_path}/{image_id_cam}.png', cam.astype(np.uint8))

            sys.stdout.write(
                f'\r# Make CAM [{step + 1}/{length}] = {((step + 1) / length) * 100:.2f}%, ({ori_h}, {ori_w}), {hr_cams.size()}'
            )
            sys.stdout.flush()
        print()

    if args.domain == 'train_aug':
        args.domain = 'train'

    print(f"python3 inference_crf.py --experiment_name {experiment_name} --domain {args.domain}")