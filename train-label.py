import sys
import matplotlib
matplotlib.use('Agg')

from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from utils import *
from datasets import *
from SDRD import *
from loss import *

# Argument Parser Setup
parser = argparse.ArgumentParser()

###############################################################################
# Dataset Arguments
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=4, type=int)  # Set to 4 for parallelism in dataset loading
parser.add_argument('--data_dir', default=r'/path/to/dataset/', type=str)  # Change this to your dataset directory
parser.add_argument('--dataset', default=r'CODtrain', type=str)

###############################################################################
# Network Arguments
###############################################################################
parser.add_argument('--mode', default='normal', type=str)  # fix mode

###############################################################################
# Hyperparameter Arguments
###############################################################################
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--max_epoch', default=40, type=int)
parser.add_argument('--depth', default=50, type=int)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--print_ratio', default=0.2, type=float)
parser.add_argument('--tag', default='trainv1', type=str)
parser.add_argument('--domain', default='train', type=str)
parser.add_argument('--augment', default='', type=str)
parser.add_argument('--alpha', type=float, default=0.25)

flag = True

if __name__ == '__main__':
    # Argument Parsing
    args = parser.parse_args()

    # Directory Setup
    log_dir = create_directory('./logs/')
    data_dir = create_directory('./data/')
    model_dir = create_directory('./models/')
    tensorboard_dir = create_directory(f'./tensorboards/{args.tag}/')

    log_path = log_dir + f'{args.tag}.txt'
    data_path = data_dir + f'{args.tag}.json'

    cam_path = f'./images/{args.tag}'
    create_directory(cam_path)
    create_directory(cam_path + '/train')
    create_directory(cam_path + '/test')
    create_directory(cam_path + '/train/colormaps')
    create_directory(cam_path + '/test/colormaps')

    # Set random seed for reproducibility
    set_seed(args.seed)
    log_func = lambda string='': log_print(string, log_path)

    # Logging initial info
    log_func(f'[i] {args.tag}')
    log_func()

    # Dataset and DataLoader setup
    train_dataset = Train_label_Datasets(args.data_dir, args.dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    val_iteration = len(train_loader)
    log_iteration = int(val_iteration * args.print_ratio)
    max_iteration = args.max_epoch * val_iteration

    log_func(f'[i] train data: {len(train_dataset):,}')
    log_func(f'[i] log_iteration: {log_iteration:,}')
    log_func(f'[i] val_iteration: {val_iteration:,}')
    log_func(f'[i] max_iteration: {max_iteration:,}')

    ###################################################################################
    # Model Setup
    ###################################################################################
    model = SDRDnet()
    param_groups = model.get_parameter_groups()
    model = model.cuda()
    model.train()

    log_func(f'[i] Total Params: {calculate_parameters(model)/1e6:.2f}M')
    log_func()

    # Model loading and saving functions
    load_model_fn = lambda: load_model(model, model_path)
    save_model_fn = lambda: save_model(model, model_path)

    ###################################################################################
    # Loss and Optimizer Setup
    ###################################################################################
    criterion = [
        SimMaxLoss(metric='cos', alpha=args.alpha).cuda(),
        SimMinLoss(metric='cos').cuda(),
        SimMaxLoss(metric='cos', alpha=args.alpha).cuda()
    ]

    optimizer = PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wd},
        {'params': param_groups[1], 'lr': 2 * args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10 * args.lr, 'weight_decay': args.wd},
        {'params': param_groups[3], 'lr': 20 * args.lr, 'weight_decay': 0},
    ], lr=args.lr, momentum=0.9, weight_decay=args.wd, max_step=max_iteration)

    ###################################################################################
    # Training Setup
    ###################################################################################
    train_timer = Timer()
    train_meter = Average_Meter(['loss', 'positive_loss', 'negative_loss'])
    writer = SummaryWriter(tensorboard_dir)

    for epoch in range(args.max_epoch):
        loop = tqdm(train_loader, total=val_iteration)
        optimizer.zero_grad()

        for iteration, (images, tokens) in enumerate(loop, 1):
            images = images.cuda()
            tokens = tokens.cuda()

            # Forward pass
            fg_feats, bg_feats, ccam = model(tokens)

            # Loss computation
            loss1 = criterion[0](fg_feats)
            loss2 = criterion[1](bg_feats, fg_feats)
            loss3 = criterion[2](bg_feats)
            loss = loss1 + loss2 + loss3

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Check if flag is set
            if epoch == 0 and iteration == 10:
                flag = check_positive(ccam)
                print(f"Is Negative: {flag}")
            if flag:
                ccam = 1 - ccam

            # Update metrics
            train_meter.add({
                'loss': loss.item(),
                'positive_loss': loss1.item() + loss3.item(),
                'negative_loss': loss2.item(),
            })

            ###################################################################################
            # Logging
            ###################################################################################
            if (iteration + 1) % 1 == 0:
                loop.set_description(f'Epoch: {epoch+1}/{args.max_epoch}')

            if (iteration + 1) % 40 == 0:
                loss, positive_loss, negative_loss = train_meter.get(clear=True)
                learning_rate = float(get_learning_rate_from_optimizer(optimizer))

                # Log data
                data = {
                    'epoch': epoch + 1,
                    'max_epoch': args.max_epoch,
                    'iteration': iteration + 1,
                    'learning_rate': learning_rate,
                    'loss': loss,
                    'positive_loss': loss1,
                    'negative_loss': loss2,
                    'time': train_timer.tok(clear=True),
                }

                log_func(f'[i]\tEpoch[{epoch+1:,}/{args.max_epoch:,}],\titeration={iteration + 1:,},\tlearning_rate={learning_rate:.6f},\tloss={loss:.4f},\tpositive_loss={positive_loss:.4f},\tnegative_loss={negative_loss:.4f},\ttime={data["time"]:.0f}sec')

                # TensorBoard logging
                writer.add_scalar('Train/loss', loss, iteration)
                writer.add_scalar('Train/learning_rate', learning_rate, iteration)

        ###################################################################################
        # Save Model Checkpoint
        ###################################################################################
        if (epoch+1) % 1 == 0:
            model_path = model_dir + f'{args.tag}-{epoch+1}.pth'
            save_model(model, model_path)
            log_func(f'[i] model saved')

    print(args.tag)