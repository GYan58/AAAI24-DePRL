from Models import Alex as Ax
from Models import ResNet as Re
from Models import VGG as Vg
from Models import DNN as Dn
from Settings import *

import warnings
warnings.filterwarnings("ignore")

def load_Model(Model, Name, Save):
    Model = None        
    if Model== "dnn":
        if Name == "har":
            Model = Dn.dnn_har()

    if Model == "alex":
        if Name == "fmnist":
            Model = Ax.alex_fmnist()

    if Model == "vgg":
        if Name == "cifar10":
            Model = Vg.vgg_cifar10()

    if Model == "resnet":
        if Name == "cifar100":
            Model = Re.resnet_cifar100()
    
    return Model

#---------------------------------------------------------------------------
def load_data_har(user_id):
    coll_class = []
    coll_label = []
    total_class = 0
    NUM_OF_CLASS = 5
    DIMENSION_OF_FEATURE = 900
    class_set = ['Call', 'Hop', 'typing', 'Walk', 'Wave']

    for class_id in range(NUM_OF_CLASS):
        read_path = HARBoxRoot + str(user_id) + Symbol + str(class_set[class_id]) + '_train' + '.txt'
        if os.path.exists(read_path):
            temp_original_data = np.loadtxt(read_path)
            temp_reshape = temp_original_data.reshape(-1, 100, 10)
            temp_coll = temp_reshape[:, :, 1:10].reshape(-1, DIMENSION_OF_FEATURE)
            count_img = temp_coll.shape[0]
            temp_label = class_id * np.ones(count_img)
            coll_class.extend(temp_coll)
            coll_label.extend(temp_label)
            total_class += 1
    coll_class = np.array(coll_class)
    coll_label = np.array(coll_label)
    return coll_class, coll_label, DIMENSION_OF_FEATURE, total_class


def get_harbox():
    NUM_OF_TOTAL_USERS = 120
    X_Trains = []
    X_Tests = []
    Y_Trains = []
    Y_Tests = []
    for current_user_id in range(1, NUM_OF_TOTAL_USERS + 1):
        x_coll, y_coll, dimension, num_of_class = load_data_har(current_user_id)
        Tsize = int(len(x_coll) * 0.2) + 1
        x_train, x_test, y_train, y_test = train_test_split(x_coll, y_coll, test_size=Tsize, random_state=0)
        X_Trains += list(x_train)
        X_Tests += list(x_test)
        Y_Trains += list(y_train)
        Y_Tests += list(y_test)

    print("* Train and Test Size:",len(X_Trains),len(X_Tests))

    train_temp = []
    test_temp = []
    for i in range(len(X_Trains)):
        train_temp.append([[X_Trains[i]]])
    for i in range(len(X_Tests)):
        test_temp.append([[X_Tests[i]]])

    X_Trains = np.array(train_temp, dtype=float)
    X_Tests = np.array(test_temp, dtype=float)
    Y_Trains = np.array(Y_Trains, dtype=int)
    Y_Tests = np.array(Y_Tests, dtype=int)

    return X_Trains, Y_Trains, X_Tests,  Y_Tests

# --------------------------------------------------------------------------

def get_cifar10():
    data_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
    data_test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
    TrainX, TrainY = data_train.data.transpose((0, 3, 1, 2)), np.array(data_train.targets)
    TestX, TestY = data_test.data.transpose((0, 3, 1, 2)), np.array(data_test.targets)
    return TrainX, TrainY, TestX, TestY

def get_cifar100():
    data_train = torchvision.datasets.CIFAR100(root="./data", train=True, download=True)
    data_test = torchvision.datasets.CIFAR100(root="./data", train=False, download=True)
    TrainX, TrainY = data_train.data.transpose((0, 3, 1, 2)), np.array(data_train.targets)
    TestX, TestY = data_test.data.transpose((0, 3, 1, 2)), np.array(data_test.targets)
    return TrainX, TrainY, TestX, TestY

def get_fmnist():
    data_train = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True)
    data_test = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True)
    TrainX, TrainY = data_train.train_data.numpy().reshape(-1, 1, 28, 28) / 255, np.array(data_train.targets)
    TestX, TestY = data_test.test_data.numpy().reshape(-1, 1, 28, 28) / 255, np.array(data_test.targets)
    return TrainX, TrainY, TestX, TestY

class Addblur(object):

    def __init__(self, blur="Gaussian"):
        self.blur = blur

    def __call__(self, img):
        if self.blur == "normal":
            img = img.filter(ImageFilter.BLUR)
            return img
        if self.blur == "Gaussian":
            img = img.filter(ImageFilter.GaussianBlur)
            return img
        if self.blur == "mean":
            img = img.filter(ImageFilter.BoxBlur)
            return img

class AddNoise(object):
    def __init__(self, noise="Gaussian"):
        self.noise = noise
        self.density = 0.8
        self.mean = 0.0
        self.variance = 10.0
        self.amplitude = 10.0

    def __call__(self, img):

        img = np.array(img) 
        h, w, c = img.shape

        if self.noise == "pepper":
            Nd = self.density
            Sd = 1 - Nd
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd])  # 生成一个通道的mask
            mask = np.repeat(mask, c, axis=2) 
            img[mask == 2] = 0
            img[mask == 1] = 255 

        if self.noise == "Gaussian":
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
            N = np.repeat(N, c, axis=2)
            img = N + img
            img[img > 255] = 255

        img = Image.fromarray(img.astype('uint8')).convert('RGB')

        return img

# ---------------------------------------------------------------------------------------------
class split_image_data(object):
    def __init__(self, dataset, labels, workers, balance=True, isIID=True, alpha=0.0, dproportion=None):
        Perts = []
        self.Dataset = dataset
        self.Labels = labels
        self.workers = workers
        self.DirichRVs = []
        self.DirichCount = 0
        self.GProportions = dproportion

        if alpha == 0 and not isIID:
            print("* Error...")

        if balance:
            for i in range(workers):
                Perts.append(1 / workers)
        else:
            Sum = workers * (workers + 1) / 2
            SProb = 0
            for i in range(workers - 1):
                prob = int((i + 1) / Sum * 10000) / 10000
                SProb += prob
                Perts.append(prob)

            Left = 1 - SProb
            Perts.append(Left)
            bfrac = 0.1 / workers
            for i in range(len(Perts)):
                Perts[i] = Perts[i] * 0.9 + bfrac

        if not isIID and alpha > 0:
            if dproportion == None:
                self.partitions = self.__getDirichlet__(labels, Perts, seed, alpha)
            else:
                self.partitions = self.get_new_batch(labels, Perts, seed, alpha)
        if isIID:
            self.partitions = []
            rng = rd.Random()
            data_len = len(labels)
            indexes = [x for x in range(0, data_len)]
            rng.shuffle(indexes)
            for frac in Perts:
                part_len = int(frac * data_len)
                self.partitions.append(indexes[0:part_len])
                indexes = indexes[part_len:]

    def __getDirichlet__(self, data, psizes, seed, alpha):
        n_nets = len(psizes)
        K = len(np.unique(self.Labels))
        labelList = np.array(data)
        min_size = 0
        N = len(labelList)

        net_dataidx_map = {}
        idx_batch = []
        while min_size < K:
            idx_batch = [[] for _ in range(n_nets)]
            for k in range(K):
                idx_k = np.where(labelList == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            net_dataidx_map[j] = idx_batch[j]

        net_cls_counts = {}

        for net_i, dataidx in net_dataidx_map.items():
            unq, unq_cnt = np.unique(labelList[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp

        local_sizes = []
        for i in range(n_nets):
            local_sizes.append(len(net_dataidx_map[i]))
        local_sizes = np.array(local_sizes)
        weights = local_sizes / np.sum(local_sizes)

        return idx_batch
        
        
    def get_new_batch(self, data, psizes, seed, alpha):
        n_nets = len(psizes)
        K = len(np.unique(self.Labels))
        labelList = np.array(data)
        min_size = 0
        N = len(labelList)

        net_dataidx_map = {}
        
        idx_batch = [[] for _ in range(n_nets)]
        for k in range(K):
            idx_k = np.where(labelList == k)[0]
            proportions = self.GProportions[k]
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]

        for j in range(n_nets):
            net_dataidx_map[j] = idx_batch[j]

        net_cls_counts = {}

        for net_i, dataidx in net_dataidx_map.items():
            unq, unq_cnt = np.unique(labelList[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp

        local_sizes = []
        for i in range(n_nets):
            local_sizes.append(len(net_dataidx_map[i]))
        local_sizes = np.array(local_sizes)
        weights = local_sizes / np.sum(local_sizes)

        return idx_batch

    def get_splits(self):
        clients_split = []
        for i in range(self.workers):
            IDx = self.partitions[i]
            if len(IDx) < 10:
                IDx += [1,2,3,4,5]
            Ls = self.Labels[IDx]
            Ds = self.Dataset[IDx]

            Xs = []
            Ys = []
            Datas = {}
            for k in range(len(Ls)):
                L = Ls[k]
                D = Ds[k]
                if L not in Datas.keys():
                    Datas[L] = [D]
                else:
                    Datas[L].append(D)

            Kys = list(Datas.keys())
            Kl = len(Kys)
            CT = 0
            k = 0
            while CT < len(Ls):
                Id = Kys[k % Kl]
                k += 1
                if len(Datas[Id]) > 0:
                    Xs.append(Datas[Id][0])
                    Ys.append(Id)
                    Datas[Id] = Datas[Id][1:]
                    CT += 1

            clients_split += [(np.array(Xs), np.array(Ys))]
            del Xs, Ys
            gc.collect()

        n_labels = len(np.unique(self.Labels))

        return clients_split


# ---------------------------------------------------------------------------------
def get_train_data_transforms(name, aug=False, blur=False, noise=False, normal=False):
    Ts = [transforms.ToPILImage()]
    if name == "fmnist":
        Ts.append(transforms.Resize((32, 32)))

    if aug == True and name == "cifar10":
        Ts.append(transforms.RandomCrop(32, padding=4))
        Ts.append(transforms.RandomHorizontalFlip())

    if blur == True:
        Ts.append(Addblur())

    if noise == True:
        Ts.append(AddNoise())

    Ts.append(transforms.ToTensor())

    if normal == True:
        if name == "fmnist":
            Ts.append(transforms.Normalize((0.1307,), (0.3081,)))
        if name == "cifar10":
            Ts.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        if name == "cifar100":
            Ts.append(transforms.Normalize((0.5071, 0.4867, 0.4480), (0.2675, 0.2565, 0.2761)))

    return transforms.Compose(Ts)


def get_test_data_transforms(name, normal=False):
    transforms_eval_F = {
        'fmnist': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]),
        'cifar10': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ]),
        'cifar100': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ]),
    }

    transforms_eval_T = {
        'fmnist': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'cifar10': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        'cifar100': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4480), (0.2675, 0.2565, 0.2761))
        ]),
    }

    if normal == False:
        return transforms_eval_F[name]
    else:
        return transforms_eval_T[name]


# ------------------------------------------------------------------
class CustomImageDataset(Dataset):
    def __init__(self, inputs, labels, transforms=None):
        assert inputs.shape[0] == labels.shape[0]
        self.inputs = torch.Tensor(inputs)
        self.labels = torch.Tensor(labels).long()
        self.transforms = transforms

    def __getitem__(self, index):
        img, label = self.inputs[index], self.labels[index]

        if self.transforms is not None:
            img = self.transforms(img)

        return (img, label)

    def __len__(self):
        return self.inputs.shape[0]


def get_loaders(Name, n_clients=20, isiid=False, alpha=0.1, aug=False, noise=False, blur=False, normal=False,dshuffle=True, batchsize=128):
    TrainX, TrainY, TestX, TestY = [], [], [], []
    if Name == "fmnist":
        TrainX, TrainY, TestX, TestY = get_fmnist()
    if Name == "cifar10":
        TrainX, TrainY, TestX, TestY = get_cifar10()
    if Name == "cifar100":
        TrainX, TrainY, TestX, TestY = get_cifar100()
    if Name == "har":
        TrainX, TrainY, TestX, TestY = get_harbox()
    
    if Name != "har":
        transforms_train = get_train_data_transforms(Name, aug, blur, noise, normal)
        transforms_eval = get_test_data_transforms(Name, normal)
    else:
        transforms_train = None
        transforms_eval = None

    SPL1 = split_image_data(TrainX, TrainY, n_clients, True, isiid, alpha)
    trsplits = SPL1.get_splits()
    proportions = SPL1.GProportions
    SPL2 = split_image_data(TestX, TestY, n_clients, True, isiid, alpha, proportions)
    tesplits = SPL2.get_splits()

    client_trloaders = []
    client_teloaders = []
    
    for x, y in trsplits:
        client_trloaders.append(
            torch.utils.data.DataLoader(CustomImageDataset(x, y, transforms_train), batch_size=batchsize, shuffle=dshuffle))

    for x, y in tesplits:
        client_teloaders.append(
            torch.utils.data.DataLoader(CustomImageDataset(x, y, transforms_eval), batch_size=batchsize, shuffle=False))

    train_loader = torch.utils.data.DataLoader(CustomImageDataset(TrainX, TrainY, transforms_train), batch_size=2000,shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(CustomImageDataset(TestX, TestY, transforms_eval), batch_size=2000,shuffle=False, num_workers=2)

    return client_trloaders, client_teloaders, train_loader, test_loader


#--------------------------------
def genRandTopo(Client,P=0.25):
    Mat = []
    for i in range(Client):
        M = []
        for j in range(Client):
            M.append(0)
        Mat.append(M)

    for i in range(Client):
        for j in range(Client):
            if j > i:
                prob = np.random.rand()
                if prob <= P:
                    Mat[i][j] = 1
                    Mat[j][i] = 1
    return Mat

def genRingTopo(Client):
    Mat = []
    for i in range(Client):
        M = []
        for j in range(Client):
            M.append(0)
        Mat.append(M)

    for i in range(Client):
        f = i - 1
        s = (i + 1) % Client
        if f < 0:
            f += Client
        Mat[i][f] = 1
        Mat[i][s] = 1
    return Mat

def genStarTopo(Client):
    Mat = []
    for i in range(Client):
        M = []
        for j in range(Client):
            M.append(0)
        Mat.append(M)

    for i in range(Client):
        if i >= 0:
            Mat[i][0] = 1
            Mat[0][i] = 1
    return Mat
    
def genFullTopo(Client):
    Mat = []
    for i in range(Client):
        M = []
        for j in range(Client):
            M.append(1)
        Mat.append(M)

    for i in range(Client):
        Mat[i][i] = 0
    return Mat


class getTopo:
    def __init__(self, Clients, P=0.25, Type="Random"):
        self.Clients = Clients
        self.P = P
        self.Type = Type
        self.Topo = None
        self.Neibs = {}
        self.updateTopo()
    
    def updateTopo(self):
        if self.Type == "Random":
            self.Topo = genRandTopo(self.Clients,self.P)
        if self.Type == "Ring":
            self.Topo = genRingTopo(self.Clients)
        if self.Type == "Star":
            self.Topo = genStarTopo(self.Clients)
        if self.Type == "Full":
            self.Topo = genFullTopo(self.Clients)
        
        self.Neibs = {}
        for i in range(len(self.Topo)):
            Ns = self.Topo[i]
            GN = [i]
            for j in range(len(Ns)):
                if Ns[j] == 1:
                    GN.append(j)
            self.Neibs[i] = list(np.unique(GN))

    def reqTopo(self,Id):
        return self.Neibs[Id]


#-----------------------------------------------------------
def init_mask(Params, sparsity):
    mask = {}
    for ky in Params.keys():
        mask[ky] = torch.zeros_like(Params[ky])
        dense_numel = int((1 - sparsity[ky]) * torch.numel(mask[ky]))
        if dense_numel > 0:
            temp = mask[ky].view(-1)
            perm = torch.randperm(len(temp))
            perm = perm[:dense_numel]
            temp[perm] = 1
    return mask

def cal_sparsity(Params, sparse=0.5, distribution = "uniform", tabu = []):
    erk_power_scale = 0.1
    sparsity = {}
    
    if distribution == "uniform":
        for ky in Params.keys():
            if ky not in tabu:
                sparsity[ky] = 1 - sparse
            else:
                sparsity[ky] = 0
        return sparsity
    
    total_params = 0
    for ky in Params.keys():
        total_params += Params[ky].numel()
    
    is_epsilon_valid = False
    dense_layer = []
    density = sparse
    while not is_epsilon_valid:
        divisor = 0
        rhs = 0
        raw_prob = {}
        for ky in Params.keys():
            if ky in tabu:
                dense_layers.add(ky)
            n_param = np.prod(Params[ky].shape)
            n_zeros = n_param * (1 - density)
            n_ones = n_param * density
            
            if ky in dense_layer:
                rhs -= n_zeros
            else:
                rhs += n_ones
                raw_prob[ky] = (np.sum(Params[ky].shape) / np.prod(Params[ky].shape)) ** erk_power_scale
                divisor += raw_prob[ky] * n_param
            
            epsilon = rhs / divisor
            
            max_prob = max(list(raw_prob.values()))
            max_prob_one = max_prob * epsilon
            if max_prob_one > 1:
                is_epsilon_valid = False
                for mask_name, mask_raw_prob in raw_prob.items():
                    if mask_raw_prob == max_prob:
                        dense_layer.append(mask_name)
            else:
                is_epsilon_valid = True
        
        for ky in Params.keys():
            if ky in dense_layer:
                sparsity[ky] = 0
            else:
                sparsity[ky] = (1 - epsilon * raw_prob[ky])
    return sparsity


def plusParas(P1,P2,Fac=1):
    Res = cp.deepcopy(P1)
    for ky in P2.keys():
        Res[ky] = P1[ky] + P2[ky] * Fac
    return Res
        
def minusParas(P1,P2,Fac=1):
    Res = cp.deepcopy(P1)
    for ky in P2.keys():
        Res[ky] = P1[ky] - P2[ky] * Fac
    return Res
    
def mulpyParas(P,Fac=1):
    Res = cp.deepcopy(P)
    for ky in P.keys():
        Res[ky] = P[ky] * Fac
    return Res

def indexParas(P1,P2):
    Res = cp.deepcopy(P2)
    for ky in P1.keys():
        Res[ky] = ((P1[ky] > 0) + (P1[ky] < 0)) * P2[ky]
    return Res

def minusParas_layer(P1,P2,Fac=1):
    Res = {}
    for ky in P2.keys():
        chk = checkKey(ky)
        if chk == False:  
            Res[ky] = cp.deepcopy(P1[ky] - P2[ky] * Fac)
    return Res


def avgParas(Paras, Lens):
    Res = cp.deepcopy(Paras[0])
    Sum = np.sum(Lens)
    for ky in Res.keys():
        Mparas = 0
        for i in range(len(Paras)):
            Pi = Lens[i] / Sum
            Mparas += Paras[i][ky] * Pi
        Res[ky] = Mparas
    return Res
    
def avgEleParas(Paras, Lens, BPara=None):
    Res = cp.deepcopy(Paras[0]) # {} # 
    for ky in Paras[0].keys():
        Mparas = 0
        Mask = 0
        BMask = 1
        if BPara != None:
            BMask = (BPara[ky] > 0) + (BPara[ky] < 0)
        for i in range(len(Paras)):
            Mask += ((Paras[i][ky] > 0) + (Paras[i][ky] < 0)) * Lens[i]
            Mparas += Paras[i][ky] * Lens[i]
        Mask = Mask + (Mask == 0) * 0.000001
        Res[ky] = Mparas / Mask * BMask
    return Res
    
    
def avgEleParas_Grad(Paras, Lens, BPara=None):
    Res = {}
    for ky in Paras[0].keys():
        Mparas = 0
        Mask = 0
        BMask = 1
        if BPara != None:
            BMask = (BPara[ky] > 0) + (BPara[ky] < 0)
        for i in range(len(Paras)):
            Mask += ((Paras[i][ky] > 0) + (Paras[i][ky] < 0)) * Lens[i]
            Mparas += Paras[i][ky] * Lens[i]
        Mask = Mask + (Mask == 0) * 0.000001
        Res[ky] = Mparas / Mask * BMask
    return Res
    
    
def avgMaskParas(Paras, Lens, BaseMask):
    Res = cp.deepcopy(Paras[0])
    for ky in Res.keys():
        Mparas = 0
        Mask = 0
        BMask = 1
        if ky in BaseMask.keys():
            BMask = BaseMask[ky]
        for i in range(len(Paras)):
            Mask += ((Paras[i][ky] > 0) + (Paras[i][ky] < 0)) * Lens[i]
            Mparas += Paras[i][ky] * Lens[i]
        Mask = Mask + (Mask == 0) * 0.000001
        Res[ky] = Mparas / Mask * BMask
    return Res

def normParas(P1,P2):
    Ns = 0
    Ps1 = cp.deepcopy(P1)
    Ps2 = cp.deepcopy(P2)
    for ky in P1.keys():
        if "bias" in ky or "weight" in ky:
            V1 = Ps1[ky].cpu().detach().numpy().reshape(-1)
            V2 = Ps2[ky].cpu().detach().numpy().reshape(-1)
            Nm = np.linalg.norm(V1 - V2, ord=2) ** 2
            Ns += Nm
    return np.sqrt(Ns)

    
def maskParas(Paras, Mask):
    Res = cp.deepcopy(Paras)
    for ky in Paras.keys():
        if ky in Mask.keys():
            Res[ky] = Paras[ky] * Mask[ky]
    return Res
    
def combParas(Paras1,Paras2):
    Res  = {}
    for ky in Paras1.keys():
        Res[ky] = Paras1[ky] + (Paras1[ky] == 0) * Paras2[ky]
    return Res


def avgEleParas_ours(Past, Now, W_past=0.1):
    Res = cp.deepcopy(Past)
    for ky in Res.keys():
        Idx = (Now[ky] > 0) + (Now[ky] < 0)
        Res[ky] = Idx * Past[ky] * W_past + Now[ky] * (1 - W_past)
    return Res


def imageSim(V1,V2):
    N1 = np.linalg.norm(V1, ord=2)
    N2 = np.linalg.norm(V2, ord=2)
    Sim = np.dot(V1, V2) / N1 / N2
    return (Sim + 1) / 2

def checkKey(Key):
    Base = ["classifier","out"]
    In = False
    for ky in Base:
        if ky in Key:
            In = True
    return In

def statParas(Paras):
    num_nonzero = 0
    num_sum = 0
    for ky in Paras.keys():
        PNow = Paras[ky].cpu().detach().numpy().reshape(-1)
        num_nonzero += np.sum(PNow > 0) + np.sum(PNow < 0)
        num_sum += len(PNow)
    
    return num_nonzero, num_sum


def statMasks(Mask):
    Nums = []
    for ky in Mask.keys():
        Nums.append(torch.sum(Mask[ky]).item())
    print("# Mask:", np.sum(Nums), Nums)
    
    
def getGCEr(GPara, Paras):
    Gs = []
    for i in range(len(Paras)):
        Ns = 0
        Ps1 = cp.deepcopy(GPara)
        Ps2 = cp.deepcopy(Paras[i])
        for ky in Ps1.keys():
            chk = checkKey(ky)
            if chk == False:
                if "bias" in ky or "weight" in ky:
                    V1 = Ps1[ky].cpu().detach().numpy().reshape(-1)
                    V2 = Ps2[ky].cpu().detach().numpy().reshape(-1)
                    Nm = np.linalg.norm(V1 - V2, ord=2) ** 2
                    Ns += Nm
        get = np.sqrt(Ns)
        Gs.append(get)
     
    return Gs


def genConfig(Algorithm):
    
    if Algorithm == "FedAvg":
        AggM = "FedAvg"
        GlobalTrainM = "syn"
        LocalTrainM = "way1"
        CommType = "Full"
        return AggM, CommType, GlobalTrainM, LocalTrainM
        
    if Algorithm == "FedRep":
        AggM = "FedAvg"
        GlobalTrainM = "syn"
        LocalTrainM = "way2"
        CommType = "Layer"
        return AggM, CommType, GlobalTrainM, LocalTrainM

    if Algorithm == "Ditto":
        AggM = "Ditto"
        GlobalTrainM = "syn"
        LocalTrainM = "way3"
        CommType = "Full"
        return AggM, CommType, GlobalTrainM, LocalTrainM
        
    if Algorithm == "FedRoD":
        AggM = "Rod"
        GlobalTrainM = "syn"
        LocalTrainM = "way5"
        CommType = "Full"
        return AggM, CommType, GlobalTrainM, LocalTrainM
    
    if Algorithm == "D-PSGD":
        AggM = "D-PSGD"
        GlobalTrainM = "syn"
        LocalTrainM = "way1"
        CommType = "Full"
        return AggM, CommType, GlobalTrainM, LocalTrainM
        
    if Algorithm == "DisPFL":
        AggM = "D-Mask"
        GlobalTrainM = "syn"
        LocalTrainM = "way6"
        CommType = "Full"
        return AggM, CommType, GlobalTrainM, LocalTrainM
        
    if Algorithm == "DePRL":
        AggM = "D-PSGD"
        GlobalTrainM = "syn"
        LocalTrainM = "way2"
        CommType = "Layer"
        return AggM, CommType, GlobalTrainM, LocalTrainM

