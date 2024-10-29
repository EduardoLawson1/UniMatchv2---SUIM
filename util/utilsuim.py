import numpy as np
import logging
import os
import cv2
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.utils import save_image


def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


"""def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    #print(f"output shape: {output.shape}")
    #print(f"target shape: {target.shape}")
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target"""
def intersectionAndUnion(output, target, K, ignore_index=255):
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(-1).copy()
    target = target.reshape(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


logs = set()


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger



""" Funções novas exclusivas para o SUIM"""
def evaluate(model, valloader, eval_mode, cfg):
    model.eval()
    all_preds = []
    all_masks = []

    for i, batch in enumerate(valloader):
        image, target = batch[:2]  # Supondo que extras são ignorados por enquanto
        
        with torch.no_grad():
            output = model(image.cuda())
        
        pred = output.argmax(dim=1).cpu().numpy()
        target = target.numpy()

        if pred.shape != target.shape:
            #print(f"Shape mismatch: pred {pred.shape}, target {target.shape}")
            pred = resize_or_crop(pred, target.shape)

        all_preds.append(pred)
        all_masks.append(target)
    
    mIoU, iou_class = calculate_metrics(all_preds, all_masks, cfg['nclass'])
    
    return mIoU, iou_class

def resize_or_crop(pred, target_shape):
    resized_pred = np.resize(pred, target_shape)
    return resized_pred

def calculate_metrics(preds, masks, nclass):
    total_intersection = np.zeros(nclass)
    total_union = np.zeros(nclass)
    total_target = np.zeros(nclass)
    
    for pred, mask in zip(preds, masks):
        intersection, union, target = intersectionAndUnion(pred, mask, nclass)
        total_intersection += intersection
        total_union += union
        total_target += target
    
    iou_class = total_intersection / total_union
    mIoU = np.mean(iou_class)
    
    return mIoU, iou_class

"""função para salvar máscaras"""
def save_masks_as_images(masks, output_dir, prefix='mask'):
    os.makedirs(output_dir, exist_ok=True)
    for i, mask in enumerate(masks):
        # Movendo a máscara de volta para a memória da CPU antes de convertê-la em um array numpy
        mask_cpu = mask.cpu()
        # Convertendo a máscara para o tipo de dados uint8 (byte) e multiplicando por 255 para manter a faixa de valores
        mask_byte = (mask_cpu * 255).clamp(0, 255).byte()
        # Convertendo a máscara para numpy array e em seguida para imagem PIL
        mask_img = Image.fromarray(mask_byte.numpy())
        mask_img.save(os.path.join(output_dir, f"{prefix}_{i}.png"))



def label_to_color(label_mask):
    
    #Converte uma máscara de rótulos de classe inteiros para uma máscara de cor RGB.
    
    label_to_color = {
        0: (0, 0, 0),       # Background waterbody
        1: (0, 0, 255),     # Human divers
        2: (0, 255, 0),     # Plants/sea-grass
        3: (0, 255, 255),   # Wrecks/ruins
        4: (255, 0, 0),     # Robots/instruments
        5: (255, 0, 255),   # Reefs and invertebrates
        6: (255, 255, 0),   # Fish and vertebrates
        7: (255, 255, 255)  # Sand/sea-floor (& rocks)
    }
    
    rgb_mask = np.zeros((label_mask.shape[0], label_mask.shape[1], 3), dtype=np.uint8)
    for label, color in label_to_color.items():
        rgb_mask[label_mask == label] = color
    
    return rgb_mask

def ensure_valid_filename(filename, default_extension='.png'):
    
    #Garante que o nome do arquivo tenha uma extensão válida.
    
    name, ext = os.path.splitext(filename)
    if not ext:
        return name + default_extension
    return filename


def save_predictions_as_images(predictions, output_dir, original_images=None, filenames=None):
    """Salva as predições como imagens coloridas e, opcionalmente, as imagens originais.
    
    :param predictions: Tensor de predições (B, C, H, W) ou lista de tensores
    :param output_dir: Diretório de saída para salvar as imagens
    :param original_images: Tensor de imagens originais (B, C, H, W) ou lista de tensores (opcional)
    :param filenames: Lista de nomes de arquivos correspondentes ou tensor (opcional)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Converter predictions para lista se for um tensor
    if isinstance(predictions, torch.Tensor):
        predictions = [pred for pred in predictions]
    
    # Preparar filenames
    if filenames is None:
        filenames = [f"image_{i}.png" for i in range(len(predictions))]
    elif isinstance(filenames, torch.Tensor):
        filenames = [f"image_{i}.png" for i in range(len(filenames))]
    elif isinstance(filenames, str):
        filenames = [filenames]
    
    # Garantir que todos os nomes de arquivo tenham extensões válidas
    filenames = [ensure_valid_filename(f) for f in filenames]
    
    # Converter original_images para lista se for um tensor
    if isinstance(original_images, torch.Tensor):
        original_images = [img for img in original_images]
    
    #print(f"Number of predictions: {len(predictions)}")
    #print(f"Number of filenames: {len(filenames)}")
    if original_images:
        print(f"Number of original images: {len(original_images)}")
    
    for i, (pred, filename) in enumerate(zip(predictions, filenames)):
        #print(f"Processing prediction {i+1}/{len(predictions)}")
        #print(f"Prediction shape: {pred.shape}")
        #print(f"Filename: {filename}")
        
        # Converte as predições para rótulos de classe
        _, label_mask = torch.max(pred, dim=0)
        label_mask = label_mask.cpu().numpy()
        
        # Converte os rótulos para cores
        rgb_mask = label_to_color(label_mask)
        
        # Gera um nome de arquivo válido
        output_filename = ensure_valid_filename(os.path.basename(filename))
        
        # Salva a imagem da predição
        img = Image.fromarray(rgb_mask)
        pred_output_path = os.path.join(output_dir, f"pred_{output_filename}")
        img.save(pred_output_path)
        #print(f"Saved prediction image to: {pred_output_path}")
        
        # Salva a imagem original, se fornecida
        if original_images:
            original_img = original_images[i]
            if isinstance(original_img, torch.Tensor):
                # Normalize os valores para o intervalo [0, 1]
                original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())
                # Converta para uint8
                original_img = (original_img * 255).byte().cpu().numpy().transpose(1, 2, 0)
            elif isinstance(original_img, np.ndarray):
                if original_img.dtype != np.uint8:
                    original_img = (original_img * 255).astype(np.uint8)
            
            original_img = Image.fromarray(original_img)
            original_output_path = os.path.join(output_dir, f"original_{output_filename}")
            original_img.save(original_output_path)
           # print(f"Saved original image to: {original_output_path}")