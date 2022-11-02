import argparse
from requests import put
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from utils.util import MetricTracker, tensor_to_PIL
from torchvision.utils import make_grid
from logger import TensorboardWriter


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=1,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=1
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    writer = TensorboardWriter(config.log_dir, logger, True)
    test_metrics = MetricTracker('loss', *[m.__name__ for m in metric_fns], writer=writer)
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()


    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output, _, _ = model(data)

            #
            # save sample images, or do something with output here
            #
            writer.set_step(i)
            p = module_metric.psnr(output=output.detach().cpu(), target=target.cpu())
            ss = module_metric.ssim(output=output.detach().cpu(), target=target.cpu())
            test_metrics.update('psnr', p)
            test_metrics.update('ssim', ss)
            writer.add_image('output', make_grid(output.cpu()))
            # image = tensor_to_PIL(output.cpu())
            # save_path = 'saved/results/' + str(i) +'.png'
            # image.save(save_path)



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
