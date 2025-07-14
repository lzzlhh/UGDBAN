import random
import numpy as np
from scipy.signal import resample
from scipy import signal
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, seq):
        for t in self.transforms:
            seq = t(seq)
        return seq


class Reshape(object):
    def __call__(self, seq):
        return seq.transpose()


class Retype(object):
    def __call__(self, seq):
        return seq.astype(np.float32)


class AddGaussian(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)


class RandomAddGaussian(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)


class Scale(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1))
        scale_matrix = np.matmul(scale_factor, np.ones((1, seq.shape[1])))
        return seq*scale_matrix


class RandomScale(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1))
            scale_matrix = np.matmul(scale_factor, np.ones((1, seq.shape[1])))
            return seq*scale_matrix


class RandomStretch(object):
    def __init__(self, sigma=0.3):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            seq_aug = np.zeros(seq.shape)
            len = seq.shape[1]
            length = int(len * (1 + (random.random()-0.5)*self.sigma))
            for i in range(seq.shape[0]):
                y = resample(seq[i, :], length)
                if length < len:
                    if random.random() < 0.5:
                        seq_aug[i, :length] = y
                    else:
                        seq_aug[i, len-length:] = y
                else:
                    if random.random() < 0.5:
                        seq_aug[i, :] = y[:len]
                    else:
                        seq_aug[i, :] = y[length-len:]
            return seq_aug


class RandomCrop(object):
    def __init__(self, crop_len=20):
        self.crop_len = crop_len

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            max_index = seq.shape[1] - self.crop_len
            random_index = np.random.randint(max_index)
            seq[:, random_index:random_index+self.crop_len] = 0
            return seq


class Normalize(object):
    def __init__(self, norm_type = "0-1"):
        assert norm_type in ["0-1","-1-1","mean-std"], f"Normalization should be '0-1','mean-std' or '-1-1', but got {norm_type}"
        self.type = norm_type
        
    def __call__(self, seq):
        if  self.type == "0-1":
            seq = (seq-seq.min())/(seq.max()-seq.min())
        elif  self.type == "-1-1":
            seq = 2*(seq-seq.min())/(seq.max()-seq.min()) + -1
        elif self.type == "mean-std":
            seq = (seq-seq.mean())/seq.std()
        return seq
def add_noise(sig,SNR): # add noise to sig
    noise = np.random.randn(*sig.shape)
    noise_var = sig.var() / np.power(10,(SNR/20))
    noise = noise /noise.std() * np.sqrt(noise_var)
    return sig + noise
class sig_process(object):
    nperseg = 30
    adjust_flag = False
    def __init__(self):
        super(sig_process,self).__init__()

    @classmethod
    def time(cls,x):
        return x

    @classmethod
    def fft(cls,x):
        x = x - np.mean(x)
        x = np.fft.fft(x)
        x = np.abs(x) / len(x)
        x = x[range(int(x.shape[0] / 2))]
        x[1:-1] = 2*x[1:-1]
        return x

    @classmethod
    def slice(cls,x):
        w = int(np.sqrt(len(x)))
        img = x[:w**2].reshape(w,w)
        return img

    @classmethod
    def STFT(cls,x,verbose=False):
        while not cls.adjust_flag:

            _,_, Zxx = signal.stft(x, nperseg=cls.nperseg)
            if abs(Zxx.shape[0] - Zxx.shape[1]) < 2:
                cls.adjust_flag = True
            elif Zxx.shape[0] > Zxx.shape[1]:
                cls.nperseg -= 1
            else:
                cls.nperseg += 1
        f, t, Zxx = signal.stft(x, nperseg=cls.nperseg)
        img = np.abs(Zxx) / len(Zxx)
        if verbose:
            return f, t, img
        else:
            return img

    @classmethod
    def STFT8(cls,x,Nc=8):
        f, t, Zxx = signal.stft(x, nperseg=Nc*2-1,noverlap=None)
        img = np.abs(Zxx) / len(Zxx)
        return img
    @classmethod
    def STFT16(cls,x):
        return sig_process.STFT8(x,Nc=16)

    @classmethod
    def STFT32(cls,x):
        return sig_process.STFT8(x, Nc=32)

    @classmethod
    def STFT64(cls,x):
        return sig_process.STFT8(x, Nc=64)

    @classmethod
    def STFT128(cls,x):
        return sig_process.STFT8(x, Nc=128)

    @classmethod
    def mySTFT(cls,x,verbose=False,nperseg=256,noverlap = None):
        if not noverlap:
            noverlap = nperseg//2
        f, t, Zxx = signal.stft(x, nperseg=nperseg,noverlap=noverlap)
        img = np.abs(Zxx) / len(Zxx)
        if verbose:
            return f, t, img
        else:
            return img