import jittor as jt
from jittor import nn
import torchvision
from models.visual_stream import VisualStream
from jittor.models import resnet18 
class JittorModuleDict(nn.Module):
    def __init__(self, modules=None):
        super().__init__()
        if modules is not None:
            for key, module in modules.items():
                self.add_module(key, module)  # 用 add_module 注册子模块

    def __getitem__(self, key):
        return getattr(self, key)  # 通过属性访问子模块

class VAANet(VisualStream):
    def __init__(self,
                 snippet_duration=16,
                 sample_size=112,
                 n_classes=8,
                 seq_len=10,
                 pretrained_resnet101_path='',
                 audio_embed_size=256,
                 audio_n_segments=16,):
        super(VAANet, self).__init__(
            snippet_duration=snippet_duration,
            sample_size=sample_size,
            n_classes=n_classes,
            seq_len=seq_len,
            pretrained_resnet101_path=pretrained_resnet101_path
        )

        self.audio_n_segments = audio_n_segments
        self.audio_embed_size = audio_embed_size

        a_resnet = resnet18(pretrained=True)
        a_conv1 = nn.Conv(1, 64, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0), bias=False)
        a_avgpool = nn.Pool(kernel_size=(8, 2), op='mean')
        a_modules = [a_conv1] + list(a_resnet.children())[1:-2] + [a_avgpool]
        self.a_resnet = nn.Sequential(*a_modules)
        fc_in_features = a_resnet.fc.weight.shape[1]
        self.a_fc = nn.Sequential(
            nn.Linear(fc_in_features, self.audio_embed_size),
            nn.BatchNorm1d(self.audio_embed_size),
            nn.Tanh()
        )

        self.aa_net = JittorModuleDict({
            'conv': nn.Sequential(
                nn.Conv1d(self.audio_embed_size, 1, 1, bias=False),
                nn.BatchNorm1d(1),
                nn.Tanh(),
            ),
            'fc': nn.Linear(self.audio_n_segments, self.audio_n_segments, bias=True),
            'relu': nn.ReLU(),
        })

        self.av_fc = nn.Linear(self.audio_embed_size + self.hp['k'], self.n_classes)

    def execute(self, visual: jt.array, audio: jt.array):
        visual = visual.transpose(0, 1).contiguous()
        #visual.div_(self.NORM_VALUE).sub_(self.MEAN)
        visual = (visual / self.NORM_VALUE) - self.MEAN


        # Visual branch
        seq_len, batch, nc, snippet_duration, sample_size, _ = visual.size()
        visual = visual.view(seq_len * batch, nc, snippet_duration, sample_size, sample_size).contiguous()
        with jt.no_grad():
            F = self.resnet(visual)
            F = jt.squeeze(F, dim=2)
            F = jt.flatten(F, start_dim=2)
        F = self.conv0(F)  # [B x 512 x 16]

        Hs = self.sa_net['conv'](F)
        Hs = jt.squeeze(Hs, dim=1)
        Hs = self.sa_net['fc'](Hs)
        As = self.sa_net['softmax'](Hs)
        As = jt.mul(As, self.hp['m'])
        alpha = As.view(seq_len, batch, self.hp['m'])

        fS = jt.mul(F, jt.unsqueeze(As, dim=1).repeat(1, self.hp['k'], 1))

        G = fS.transpose(1, 2).contiguous()
        Hc = self.cwa_net['conv'](G)
        Hc = jt.squeeze(Hc, dim=1)
        Hc = self.cwa_net['fc'](Hc)
        Ac = self.cwa_net['softmax'](Hc)
        Ac = jt.mul(Ac, self.hp['k'])
        beta = Ac.view(seq_len, batch, self.hp['k'])

        fSC = jt.mul(fS, jt.unsqueeze(Ac, dim=2).repeat(1, 1, self.hp['m']))
        fSC = jt.mean(fSC, dim=2)
        fSC = fSC.view(seq_len, batch, self.hp['k']).contiguous()
        fSC = fSC.permute(1, 2, 0).contiguous()

        Ht = self.ta_net['conv'](fSC)
        Ht = jt.squeeze(Ht, dim=1)
        Ht = self.ta_net['fc'](Ht)
        At = self.ta_net['relu'](Ht)
        gamma = At.view(batch, seq_len)

        fSCT = jt.mul(fSC, jt.unsqueeze(At, dim=1).repeat(1, self.hp['k'], 1))
        fSCT = jt.mean(fSCT, dim=2)  # [bs x 512]

        # Audio branch
        bs = audio.size(0)
        audio = audio.transpose(0, 1).contiguous()
        audio = audio.chunk(self.audio_n_segments, dim=0)
        audio = jt.stack(audio, dim=0).contiguous()
        audio = audio.transpose(1, 2).contiguous()  # [16 x bs x 256 x 32]
        audio = jt.flatten(audio, start_dim=0, end_dim=1)  # [B x 256 x 32]
        audio = jt.unsqueeze(audio, dim=1)
        audio = self.a_resnet(audio)
        audio = jt.flatten(audio, start_dim=1).contiguous()
        audio = self.a_fc(audio)
        audio = audio.view(self.audio_n_segments, bs, self.audio_embed_size).contiguous()
        audio = audio.permute(1, 2, 0).contiguous()

        Ha = self.aa_net['conv'](audio)
        Ha = jt.squeeze(Ha, dim=1)
        Ha = self.aa_net['fc'](Ha)
        Aa = self.aa_net['relu'](Ha)

        fA = jt.mul(audio, jt.unsqueeze(Aa, dim=1).repeat(1, self.audio_embed_size, 1))
        fA = jt.mean(fA, dim=2)  # [bs x 256]

        # Fusion
        fSCTA = jt.cat([fSCT, fA], dim=1)
        output = self.av_fc(fSCTA)

        return output, alpha, beta, gamma
