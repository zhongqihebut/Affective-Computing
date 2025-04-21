#import torch.nn as nn
from models.vaanet import VAANet
from models.vaanet_erase import VAANetErase
from models.visual_stream import VisualStream
from models.visual_stream_w_Erase import VisualErase
import os
import jittor.nn as nn
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def generate_model(opt):
    model = VAANet(
        snippet_duration=opt.snippet_duration,
        sample_size=opt.sample_size,
        n_classes=opt.n_classes,
        seq_len=opt.seq_len,
        audio_embed_size=opt.audio_embed_size,
        audio_n_segments=opt.audio_n_segments,
        pretrained_resnet101_path=opt.resnet101_pretrained,
    )
    model = model.cuda()
    return model, model.parameters()

def generate_vaaerase_model(opt):
    model = VAANetErase(
        snippet_duration=opt.snippet_duration,
        sample_size=opt.sample_size,
        n_classes=opt.n_classes,
        seq_len=opt.seq_len,
        audio_embed_size=opt.audio_embed_size,
        audio_n_segments=opt.audio_n_segments,
        pretrained_resnet101_path=opt.resnet101_pretrained,
    )
    #model = nn.DataParallel(model, device_ids=[0, 1])
    model = model.cuda()
    return model, model.parameters()

def generate_visual_model(opt):
    model=VisualStream(
        snippet_duration=opt.snippet_duration,
        sample_size=opt.sample_size,
        n_classes=opt.n_classes,
        seq_len=opt.seq_len,
        pretrained_resnet101_path=opt.resnet101_pretrained,
    )
    #model = nn.DataParallel(model, device_ids=[0, 1])
    model=model.cuda()
    return model,model.parameters()

def generate_visual_Erase_model(opt):
    model=VisualErase(
        snippet_duration=opt.snippet_duration,
        sample_size=opt.sample_size,
        n_classes=opt.n_classes,
        seq_len=opt.seq_len,
        pretrained_resnet101_path=opt.resnet101_pretrained,
    )
    #model = nn.DataParallel(model)
    #model = nn.DataParallel(model, device_ids=[0, 1])
    model=model.cuda()
    return model, model.parameters()
