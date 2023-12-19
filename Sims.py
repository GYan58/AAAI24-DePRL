from Utils import *
from Settings import *
from scipy.stats import norm

class Client_Sim:
    def __init__(self, TrLoader, TeLoader, Model, Lr, wdecay, epoch=1, Comm="Full", dense_ratio=0.5):
        self.TrainData = cp.deepcopy(TrLoader)
        self.TestData = cp.deepcopy(TeLoader)
        self.DataName = Dataname
        self.TrLen = 0 
        self.TeLen = 0
        self.evalMX = []
        self.evalMY = []
        self.Distribution = {}
        C = 0
        for batch_id, (inputs, targets) in enumerate(self.TrainData):
            if C < 6:
                inputs, targets = inputs.to(device), targets.to(device)
                self.evalMX.append(inputs)
                self.evalMY.append(targets)
            C += 1
        
        self.Model = cp.deepcopy(Model)
        self.GModel = cp.deepcopy(Model)
        self.Wdecay = wdecay
        self.Epoch = epoch
        self.dense_ratio = dense_ratio
        sparsity = cal_sparsity(self.get_train_full_params(),self.dense_ratio)
        self.Local_Mask = cp.deepcopy(init_mask(self.get_train_full_params(),sparsity))
        self.anneal_factor = 0.5
        self.anneal_len = 20
        self.Layer_Local_Mask = False
        self.optimizer = torch.optim.SGD(self.Model.parameters(), lr=Lr, momentum=0.9, weight_decay=self.Wdecay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.96)
        self.loss_fn = nn.CrossEntropyLoss()
        self.Goptimizer = torch.optim.SGD(self.GModel.parameters(), lr=Lr, momentum=0.9, weight_decay=self.Wdecay)
        self.Gscheduler = torch.optim.lr_scheduler.StepLR(self.Goptimizer, step_size=1, gamma=0.96)
        self.Gloss_fn = nn.CrossEntropyLoss()
        self.TrLoss = 10000
        self.TrGrad = 10000
        self.Grads = None
    
    def screen_gradients(self):
        SModel = cp.deepcopy(self.Model)
        SModel.to(device)
        SModel.train()
        loss_fn = nn.CrossEntropyLoss()
        Lrnow = self.getLR()
        optimizer = torch.optim.SGD(SModel.parameters(), lr=Lrnow, momentum=0.9, weight_decay=self.Wdecay)
        
        gradient = {}
        for batch_id, (inputs, targets) in enumerate(self.TrainData):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = SModel.forward(inputs)
            optimizer.zero_grad()
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            for name, param in SModel.named_parameters():
                if name not in gradient.keys():
                    gradient[name] = param.grad
                else:
                    gradient[name] += param.grad
        return gradient
    
    def fire_mask(self, mask, weights):
        r_factor = min(0.95,self.Round / self.anneal_len)
        drop_ratio = self.anneal_factor / 2 * (1 + np.cos(r_factor * np.pi))
        new_mask = cp.deepcopy(mask)
        num_remove = {}
        
        for ky in mask.keys():
            num_non_zeros = torch.sum(mask[ky])
            num_remove[ky] = math.ceil(drop_ratio * num_non_zeros)
            temp_weights = torch.where(mask[ky] > 0,torch.abs(weights[ky]), 100000 * torch.ones_like(weights[ky]))
            x, idx = torch.sort(temp_weights.view(-1).to(device))
            new_mask[ky].view(-1)[idx[:num_remove[ky]]] = 0
        return new_mask, num_remove
    
    def regrow_mask(self, mask, num_remove, gradient=None):
        new_mask = cp.deepcopy(mask)
        for ky in mask.keys():
            temp = torch.where(mask[ky] == 0, torch.abs(gradient[ky]),-100000 * torch.ones_like(gradient[ky]))
            sort_temp, idx = torch.sort(temp.view(-1).to(device),descending=True)
            new_mask[ky].view(-1)[idx[:num_remove[ky]]] = 1
        return new_mask
    
    def update_mask(self):
        weights = self.get_train_full_params()
        gradient = self.screen_gradients()
        mask, num_remove = self.fire_mask(self.Local_Mask, weights)
        self.Local_Mask = self.regrow_mask(mask, num_remove, gradient)
        
    def get_full_params(self):
        return cp.deepcopy(self.Model.state_dict())
    
    def get_layer_params(self):
        GParas = cp.deepcopy(self.Model.state_dict())
        gparams = {}
        for ky in GParas.keys():
            chk = checkKey(ky)
            if chk == False:
                gparams[ky] = cp.deepcopy(GParas[ky])
        return gparams
    
    def get_global_full_params(self):
        return cp.deepcopy(self.GModel.state_dict())
    
    def get_global_layer_params(self):
        GParas = cp.deepcopy(self.GModel.state_dict())
        gparams = {}
        for ky in GParas.keys():
            chk = checkKey(ky)
            if chk == False:
                gparams[ky] = cp.deepcopy(GParas[ky])
        return gparams

    def transmit(self, K=None):
        TUpdate = None
        TUpdate = None
        if self.CType == "Full":
            TUpdate = self.get_full_params()
        if self.CType == "Layer":
            TUpdate = self.get_layer_params()
        return TUpdate
    
    def updateParas(self, Paras):
        LParas = cp.deepcopy(self.Model.state_dict())
        if self.CType == "Layer":
            for ky in LParas.keys():
                chk = checkKey(ky)
                if chk == False:
                    LParas[ky] = cp.deepcopy((Paras[ky] == 0) * LParas[ky] + Paras[ky])
        if self.CType == "Full":
            LParas = cp.deepcopy(Paras)
        self.Model.load_state_dict(LParas)

    def updateLR(self, lr, initGModel=False):
        self.optimizer = torch.optim.SGD(self.Model.parameters(), lr=lr, momentum=0.9, weight_decay=self.Wdecay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.96)
        if initGModel == True:
            self.GModel = cp.deepcopy(self.Model)
            self.Goptimizer = torch.optim.SGD(self.GModel.parameters(), lr=lr, momentum=0.9, weight_decay=self.Wdecay)
            self.Gscheduler = torch.optim.lr_scheduler.StepLR(self.Goptimizer, step_size=1, gamma=0.96) 
    
    def getLR(self):
        LR = self.optimizer.state_dict()['param_groups'][0]['lr']
        return LR    
    
    def updateGoParas(self,Paras,Type="full"):
        if Type == "full":
            self.GModel.load_state_dict(Paras)
        if Type == "layer":
            LParas = self.get_global_full_params()
            for ky in LParas.keys():
                chk = checkKey(ky)
                if chk == False:
                    LParas[ky] = cp.deepcopy(Paras[ky])
            self.GModel.load_state_dict(LParas)
    
    def globaltrain(self):
        self.GModel.train()
        for r in range(self.Epoch):
            for batch_id, (inputs, targets) in enumerate(self.TrainData):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.GModel(inputs)
                self.Goptimizer.zero_grad()
                loss = self.Gloss_fn(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.GModel.parameters(),10)
                self.Goptimizer.step()
        self.Gscheduler.step()  
        gc.collect()
        torch.cuda.empty_cache()
        
    
    def localtrain(self, trainway="way1", Checkupdate=False):
        self.Round += 1
        self.Model.train()
        
        if trainway == "way1":
            self.GModel = None
            BeforeParas = self.get_train_full_params()
            
            self.TrLoss = 0
            getloss = 0
            local_iters = 0
            for r in range(self.Epoch):
                for batch_id, (inputs, targets) in enumerate(self.TrainData):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = self.Model(inputs)
                    self.optimizer.zero_grad()
                    loss = self.loss_fn(outputs, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(self.Model.parameters(),10)
                    self.optimizer.step()
                    
                    lval = (loss.item()) ** 2
                    getloss += lval
                    local_iters += 1
           
            self.TrLoss = np.sqrt(getloss / local_iters) * local_iters
            AfterParas = self.get_train_full_params()
            self.Grads = minusParas(AfterParas, BeforeParas)
        
        
        if trainway == "way2":
            self.GModel = None
            BeforeParas = self.get_full_params()
            for r in range(self.Epoch):
                for batch_id, (inputs, targets) in enumerate(self.TrainData):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = self.Model(inputs)
                    self.optimizer.zero_grad()
                    loss = self.loss_fn(outputs, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(self.Model.parameters(),10)
                    self.optimizer.step()
            
                    AfterParas = self.get_full_params()
                    for ky in AfterParas.keys():
                        chk = checkKey(ky)
                        if chk == False:
                            AfterParas[ky] = cp.deepcopy(BeforeParas[ky])
                    self.Model.load_state_dict(AfterParas)
            
            BeforeParas = self.get_full_params()
            for r in range(1):
                for batch_id, (inputs, targets) in enumerate(self.TrainData):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = self.Model(inputs)
                    self.optimizer.zero_grad()
                    loss = self.loss_fn(outputs, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(self.Model.parameters(),10)
                    self.optimizer.step()
            
                    AfterParas = self.get_full_params()
                    for ky in AfterParas.keys():
                        chk = checkKey(ky)
                        if chk == True:
                            AfterParas[ky] = cp.deepcopy(BeforeParas[ky])
                    self.Model.load_state_dict(AfterParas)
        
        
        if trainway == "way3":
            GoParas = self.get_global_full_params()
            Lambda = 1
            LrNow = self.getLR()
            for batch_id, (inputs, targets) in enumerate(self.TrainData):
                    PrevParas = cp.deepcopy(self.Model.state_dict())
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = self.Model(inputs)
                    self.optimizer.zero_grad()
                    loss = self.loss_fn(outputs, targets)
                    loss.backward()
                    self.optimizer.step()
                    DGrad = minusParas(PrevParas,GoParas)
                    CurtParas = cp.deepcopy(self.Model.state_dict())
                    FParas = minusParas(CurtParas,DGrad,Lambda * LrNow)
                    self.Model.load_state_dict(FParas)
                    self.Model.train()
                 
        
        if trainway == "way4":
            self.GModel = None
            
            if self.Get_Classifier == None:
                BeforeParas = self.get_train_full_params()
                self.Get_Classifier = {}
                for ky in BeforeParas.keys():
                    chk = checkKey(ky)
                    if chk == True:
                        self.Get_Classifier[ky] = cp.deepcopy(BeforeParas[ky])
             
            if self.Get_Classifier != None:
                BeforeParas = self.get_full_params()
                for ky in self.Get_Classifier.keys():
                    BeforeParas[ky] = cp.deepcopy(self.Get_Classifier[ky])
                self.Model.load_state_dict(BeforeParas)
            
            for r in range(self.Epoch):
                for batch_id, (inputs, targets) in enumerate(self.TrainData):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = self.Model(inputs)
                    self.optimizer.zero_grad()
                    loss = self.loss_fn(outputs, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(self.Model.parameters(),10)
                    self.optimizer.step()
            
                    AfterParas = self.get_full_params()
                    if self.Get_Classifier != None:
                        for ky in self.Get_Classifier.keys():
                            AfterParas[ky] = cp.deepcopy(self.Get_Classifier[ky])
                    self.Model.load_state_dict(AfterParas)
                    self.Model.train()
                    
        
        if trainway == "way5":
            GoParas = self.get_global_full_params()
            TParas = self.get_full_params()
            for ky in TParas.keys():
                chk = checkKey(ky)
                if chk == False:
                    TParas[ky] = cp.deepcopy(GoParas[ky])
            self.Model.load_state_dict(TParas)
            self.Model.train()
            
            LrNow = self.getLR()
            ft_optimizer = torch.optim.SGD(self.Model.parameters(), lr=LrNow, momentum=0.9, weight_decay=self.Wdecay)
            for r in range(1):
                for batch_id, (inputs, targets) in enumerate(self.TrainData):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = self.Model(inputs)
                    ft_optimizer.zero_grad()
                    loss = self.loss_fn(outputs, targets)
                    loss.backward()
                    ft_optimizer.step()
                    PostParas = self.get_full_params()
                    for ky in PostParas.keys():
                        chk = checkKey(ky)
                        if chk == False:
                            PostParas[ky] = cp.deepcopy(GoParas[ky])
                    self.Model.load_state_dict(PostParas)
                    self.Model.train()

                   
        if trainway == "way6":
            self.GModel = None
            for r in range(self.Epoch):
                for batch_id, (inputs, targets) in enumerate(self.TrainData):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = self.Model(inputs)
                    self.optimizer.zero_grad()
                    loss = self.loss_fn(outputs, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(self.Model.parameters(),10)
                    self.optimizer.step()
                    
                    for name, param in self.Model.named_parameters():
                        if name in self.Local_Mask.keys():
                            param.data = cp.deepcopy(param.data * self.Local_Mask[name].to(device))
            
        self.scheduler.step()  

    def evaluate(self, loader=None, Paras=None):
        if self.TeLen <= 10:
            return None, None
        self.Model.eval()
        EModel = cp.deepcopy(self.Model)
        if Paras != None:
            EModel.load_state_dict(Paras)
        
        if loader == None:
            loader = self.TestData
            
        loss, correct, samples, iters = 0, 0, 0, 0
        loss_fn = nn.CrossEntropyLoss()
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device)
                y_ = EModel(x)
                _, preds = torch.max(y_.data, 1)
                correct += (preds == y).sum().item()
                loss += loss_fn(y_, y).item()
                samples += y_.shape[0]
                iters += 1
        torch.cuda.empty_cache()
        return correct / samples, loss / iters
        
    
    def evalModel(self, Paras=None):
        Model = cp.deepcopy(self.Model)
        if Paras != None:
            Model.load_state_dict(Paras)
        loss, correct, samples, iters = 0, 0, 0, 0
        loss_fn = nn.CrossEntropyLoss()
        with torch.no_grad():
            for i in range(len(self.evalMX)):
                x = self.evalMX[i]
                y = self.evalMY[i]
                y_ = Model(x)
                _, preds = torch.max(y_.data, 1)
                correct += (preds == y).sum().item()
                loss += loss_fn(y_, y).item()
                samples += y_.shape[0]
                iters += 1
        torch.cuda.empty_cache()
        return correct / samples, loss / iters


class Server_Sim:
    def __init__(self, TestLoader, Model, Lr, Fixlr=False, Clients=64):
        self.TestData = cp.deepcopy(TestLoader)
        self.Model = cp.deepcopy(Model)
        self.FixLr = Fixlr
        self.optimizer = torch.optim.SGD(self.Model.parameters(), lr=Lr, momentum=0.9, weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=1,gamma=0.96)
        self.loss_fn = nn.CrossEntropyLoss()
        self.LastParas = cp.deepcopy(self.Model.state_dict())
        self.RecvUpdates = []
        self.RecvLens = []
        self.MaxAccu = 0
        self.Round = 0

    def getParas(self):
        GParas = cp.deepcopy(self.Model.state_dict())
        return GParas

    def getLR(self):
        LR = self.optimizer.state_dict()['param_groups'][0]['lr']
        return LR
    
    def updateParas(self, Paras):
        UParas = cp.deepcopy(self.Model.state_dict())
        for ky in Paras.keys():
            UParas[ky] = (Paras[ky] == 0) * UParas[ky] + Paras[ky]
        self.Model.load_state_dict(UParas)
    
    def recvInfo(self, Update, Len):
        self.RecvUpdates.append(Update)
        self.RecvLens.append(Len)
    
    def synParasUpdate(self):
        self.Round += 1
        if len(self.RecvUpdates) > 2:
            self.LastParas = cp.deepcopy(self.Model.state_dict())
            globalUpdate = avgEleParas(self.RecvUpdates, self.RecvLens)
            self.updateParas(globalUpdate)
        
        self.RecvUpdates = []
        self.RecvLens = []
        if self.FixLr == False:
            self.optimizer.step()
            self.scheduler.step()
      
    def evaluate(self):
        self.Model.eval()
        loss, correct, samples, iters = 0, 0, 0, 0
        with torch.no_grad():
            for i, (x, y) in enumerate(self.TestData):
                x, y = x.to(device), y.to(device)
                y_ = self.Model(x)
                _, preds = torch.max(y_.data, 1)
                loss += self.loss_fn(y_, y).item()
                correct += (preds == y).sum().item()
                samples += y_.shape[0]
                iters += 1
        print("* Server Test:", correct, samples)
        return correct / samples, loss / iters
    
    
    def evalModel(self, Paras):
        Model = cp.deepcopy(self.Model)
        Model.load_state_dict(Paras)
        loss, correct, samples, iters = 0, 0, 0, 0
        loss_fn = nn.CrossEntropyLoss()
        with torch.no_grad():
            for i, (x, y) in enumerate(self.TestData):
                x, y = x.to(device), y.to(device)
                y_ = Model(x)
                _, preds = torch.max(y_.data, 1)
                correct += (preds == y).sum().item()
                loss += loss_fn(y_, y).item()
                samples += y_.shape[0]
                iters += 1
        torch.cuda.empty_cache()
        
        return correct / samples, loss / iters


    def saveModel(self, Path):
        torch.save(self.Model, Path)





