from Utils import *
from Sims import *

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

class PFL:
    def __init__(self, configs):
        self.DataName = configs["dname"]
        self.ModelName = configs["mname"]
        self.NClients = configs["nclients"]
        self.TopoType = configs["topo_type"]
        self.IsIID = configs["isIID"]
        self.Alpha = configs["alpha"]
        self.Aug = configs["aug"] 
        self.MaxIter = configs["iters"]
        self.LogStep = configs["logstep"]
        self.LR = configs["learning_rate"]
        self.FixLR = configs["fixlr"]
        self.GlobalLR = configs["global_lr"]
        self.WDecay = configs["wdecay"]
        self.BatchSize = configs["batch_size"]
        self.Epoch = configs["epoch"]
        self.DShuffle = configs["data_shuffle"]
        self.Normal = configs["normal"]
        self.AggMethod = configs["aggmethod"]
        self.CommType = configs["comm_type"]
        self.ConrRate = configs["conpress_rate"]
        self.GlobalTrainMethod = configs["global_train_method"]
        self.LocalTrainMethod = configs["local_train_method"]
        self.Server = None
        self.GModel = load_Model(self.ModeName,sefl.DataName)
        self.Clients = {}
        self.Client_TrLoaders = None
        self.Client_TeLoaders = None
        self.TrainLoader = None
        self.TestLoader = None
        self.UpdateIDs = [0]
        self.GetNeib = getTopo(configs["nclients"],0.25,self.TopoType)
        self.Trained = False


    def getDatas(self):
        self.Client_TrLoaders, self.Client_TeLoaders, self.TrainLoader, self.TestLoader = get_loaders(self.DataName, self.NClients, self.IsIID, self.Alpha, self.Aug, False, False, self.Normal, self.DShuffle, self.BatchSize)

    def Logging(self):
        for ky in range(self.NClients):
            accu,_ = self.Clients[ky].evaluate()

    def main(self):
        self.getDatas()

        self.Server = Server_Sim(self.TestLoader,  self.GModel, self.LR, self.FixLR, self.NClients)
        
        for c in range(self.NClients):
            self.Clients[c] = Client_Sim(self.Client_TrLoaders[c], self.Client_TeLoaders[c], self.GModel, self.LR, self.WDecay, self.Epoch, self.CommType, self.ConrRate)
        
        TotalIDs = []
        for ky in range(self.NClients):
            TotalIDs.append(ky)
        
        TotalTime = 0
        Round = 0
        
        for it in range(self.MaxIter):
            self.Trained = True
            if self.GlobalTrainMethod == "syn":
                Round = it
                
                updateIDs = TotalIDs
                
                GlobalLr = self.Server.getLR()
                GParas = self.Server.getParas()
                
                if self.AggMethod == "FedAvg":
                    for ky in updateIDs:
                        self.Clients[ky].updateParas(GParas)
                        self.Clients[ky].updateLR(GlobalLr)
                        self.Clients[ky].localtrain(self.LocalTrainMethod)
                        ET = time.time()
                        
                        LcUpdate = self.Clients[ky].transmit()
                        LcLen = self.Clients[ky].TrLen
                        self.Server.recvInfo(LcUpdate,LcLen)    
                    self.Server.synParasUpdate()
                
                if self.AggMethod == "FedAvg-FT":
                    for ky in updateIDs:
                        self.Clients[ky].updateParas(GParas)
                        self.Clients[ky].updateLR(GlobalLr)
                        self.Clients[ky].localtrain(self.LocalTrainMethod)
                        
                        LcUpdate = self.Clients[ky].transmit()
                        LcLen = self.Clients[ky].TrLen
                        self.Server.recvInfo(LcUpdate,LcLen)    
                    self.Server.synParasUpdate()
                
                if self.AggMethod == "Ditto":
                    for ky in updateIDs:
                        self.Clients[ky].updateLR(GlobalLr,initGModel=True)
                        self.Clients[ky].updateGoParas(GParas,"full")
                        self.Clients[ky].globaltrain()
                        self.Clients[ky].localtrain(self.LocalTrainMethod)
                        
                        LcUpdate = self.Clients[ky].get_global_full_params()
                        LcLen = self.Clients[ky].TrLen
                        self.Server.recvInfo(LcUpdate,LcLen)
                        
                        self.Clients[ky].GModel = None
                    
                    self.Server.synParasUpdate()
                
                if self.AggMethod == "Rod":
                    for ky in updateIDs:
                        self.Clients[ky].updateLR(GlobalLr,initGModel=True)
                        self.Clients[ky].updateGoParas(GParas,"full")
                        self.Clients[ky].localtrain(self.LocalTrainMethod)
                        self.Clients[ky].globaltrain()
                        
                        LcUpdate = self.Clients[ky].get_global_full_params()
                        LcLen = self.Clients[ky].TrLen
                        self.Server.recvInfo(LcUpdate,LcLen)
                        
                        self.Clients[ky].GModel = None
                    
                    self.Server.synParasUpdate()
                
                if self.AggMethod == "D-PSGD":
                    GetParas = {}
                    for ky in self.Clients.keys():
                        WGroup = self.GetNeib.reqTopo(ky)
                        PsNow = cp.deepcopy([])
                        LsNow = []
                        for gid in WGroup:
                            PsNow.append(self.Clients[gid].transmit())
                            LsNow.append(1)
                        GetUpdate =  avgEleParas(PsNow, LsNow)
                        GetParas[ky] = GetUpdate
                    
                    for ky in self.Clients.keys():
                        self.Clients[ky].updateLR(GlobalLr)
                        self.Clients[ky].updateParas(GetParas[ky])
                        self.Clients[ky].localtrain(self.LocalTrainMethod)
                     
                    self.Server.synParasUpdate()
                    GetParas = cp.deepcopy({})
                    torch.cuda.empty_cache()

                if self.AggMethod == "D-Mask":
                    GetParas = {}
                    for ky in self.Clients.keys():
                        WGroup = self.GetNeib.reqTopo(ky)
                        PsNow = cp.deepcopy([])
                        LsNow = []
                        for gid in WGroup:
                            PsNow.append(self.Clients[gid].transmit())
                            LsNow.append(1)
                        GetUpdate =  avgEleParas(PsNow, LsNow)
                        GetParas[ky] = GetUpdate
                        torch.cuda.empty_cache()
                    
                    for ky in self.Clients.keys():
                        self.Clients[ky].updateLR(GlobalLr)
                        self.Clients[ky].updateParas(GetParas[ky])
                        GetParas[ky] = cp.deepcopy(None)
                        self.Clients[ky].update_mask()
                        self.Clients[ky].localtrain(self.LocalTrainMethod)
                        
                        torch.cuda.empty_cache()
                    
                    self.Server.synParasUpdate()
                        
                gc.collect()
                torch.cuda.empty_cache()



if __name__ == '__main__':
    Algorithm = "DePRL"
    
    Configs = {}
    Configs['dname'] = "cifar10"
    Configs["mname"] = "vgg"
    Configs['nclients'] = 128
    Configs["epoch"] = 3
    Configs["learning_rate"] = 0.01
    Configs["iters"] = 200
    Configs['topo_type'] = "Random"
    Configs['isIID'] = False
    Configs["normal"] = True
    Configs["fixlr"] = False
    Configs["global_lr"] = True
    Configs["aug"] = False
    Configs["data_shuffle"] = True
    Configs["save_model"] = False
    Configs["fix_id"] = True
    Configs["conpress_rate"] = 0.8
    Configs["batch_size"] = 16
    Configs["wdecay"] = 1e-5
    Configs["alpha"] = 0.1
    AggM, CommType, GlobalTrainM, LocalTrainM = genConfig(Algorithm)
    Configs['comm_type'] = CommType
    Configs["aggmethod"] = AggM
    Configs["global_train_method"] = GlobalTrainM
    Configs["local_train_method"] = LocalTrainM
                
    Prototype = PFL(Configs)
    Prototype.main()
                        
