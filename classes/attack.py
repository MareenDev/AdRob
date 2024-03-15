from classes.models import RestApiCall
import numpy as np
from art.estimators.classification.blackbox import BlackBoxClassifier
from art.attacks.evasion import HopSkipJump, SignOPTAttack, SquareAttack, ZooAttack
from classes.framework import EvasionAttack


class HSJ(EvasionAttack):
    """HopSkipJump Attack - Decision-based attack
     Using the implementation of Adversarial Robustness Toolbox"""
    def __init__(self,name,apiCall,shape,targeted,norm,batchsize,verbose):
        super(HSJ, self).__init__(name,apiCall,shape,targeted,norm,batchsize,verbose)
        self.attackname ="HSJ"
        self.batchsize = batchsize
        bbClassifier = BlackBoxClassifier(predict_fn=self.APICall.predict, input_shape=shape, nb_classes=len(apiCall.getLabelEncoding()))
        self.attack = HopSkipJump(classifier=bbClassifier, batch_size=batchsize, norm=norm, targeted= targeted,verbose=verbose,
                                  max_iter=0, max_eval=1, init_eval=1 )

    def generate(self,x_ref,name,**kwargs):
        self.attack.init_eval=kwargs['eval_init']
        self.attack.max_eval = kwargs['eval_max']

        x_adv = None
        y_ref = self.APICall.predict(x = x_ref)  
        self.APICall.resetCounts()

        x_sublist = [x_ref.squeeze()]  # List of lists classifier input-data.  
        y_sublist= [np.argmax(y_ref)] # List of lists classifier output-data (classes).
        querycount_sublist = [0] # List of lists of query-counts - the sum of elements per sublist contains the total counts of API-calls for generating the last adv. ex
        
        for i in range(kwargs["iter"]):
            self.APICall.resetCounts()

            x_adv = self.attack.generate(x=x_ref, x_adv_init=x_adv, resume=True) 
            self.attack.max_iter = kwargs["iter_max"]
            
            # get label of malicious/generated data 
            y_adv=self.attack.estimator.predict(x=x_adv,batch_size=self.batchsize) 
            self.APICall.decrementCounts()  

            x_sublist    += [x_adv.squeeze()]
            y_sublist    += [np.argmax(y_adv)]
            querycount_sublist +=[self.APICall.getCounts()]
            
        self.x_list[name] = x_sublist
        self.y_list[name] = y_sublist
        self.querycount[name] = querycount_sublist


class SignOPT(EvasionAttack):
    """SignOpt Attack - Decision-based attack
     Using the implementation of Adversarial Robustness Toolbox"""
    def __init__(self,name,apiCall,shape,targeted,norm,batchsize,verbose):
        super(SignOPT, self).__init__(name,apiCall,shape,targeted,norm,batchsize,verbose)
        self.attackname ="SignOpt"
        self.bbClassifier = BlackBoxClassifier(predict_fn=apiCall.predict, input_shape=shape, nb_classes=len(apiCall.getLabelEncoding()),
                                          clip_values=(0,1))
        self.targeted = targeted
        self.batchsize = batchsize
        self.verbose = verbose
        self.APICall = apiCall

    def generate(self,x_ref,name,**kwargs):
        attack = SignOPTAttack(estimator=self.bbClassifier, targeted=self.targeted,batch_size=self.batchsize,verbose=self.verbose,
                               epsilon=kwargs['epsilon'], num_trial=kwargs['num_trial'],max_iter=kwargs['iter_max'], k=kwargs['k'],
                               query_limit=kwargs['query_limit'], alpha=kwargs['alpha'],beta=kwargs['beta'],eval_perform=kwargs['eval_perform'])
        x_adv = None
        y_ref = self.APICall.predict(x = x_ref)  
        self.APICall.resetCounts()

        x_adv = attack.generate(x=x_ref)
        y_adv=attack.estimator.predict(x=x_adv,batch_size=self.batchsize) 
        self.APICall.decrementCounts()  

        self.x_list[name] = [x_ref.squeeze(),x_adv.squeeze()]  
        self.y_list[name] = [np.argmax(y_ref),np.argmax(y_adv)]
        self.querycount[name] = [0,self.APICall.getCounts()]


class ZOO(EvasionAttack):
    """SignOpt Attack - Decision-based attack
     Using the implementation of Adversarial Robustness Toolbox"""
    def __init__(self,name,apiCall,shape,targeted,norm,batchsize,verbose):
        super(ZOO, self).__init__(name,apiCall,shape,targeted,norm,batchsize,verbose)
        self.attackname ="ZOO"
        self.bbClassifier = BlackBoxClassifier(predict_fn=apiCall.predict, input_shape=shape, nb_classes=len(apiCall.getLabelEncoding()),
                                          clip_values=(0,1))
        self.targeted = targeted
        self.batchsize = batchsize
        self.verbose = verbose
        self.APICall = apiCall

    def generate(self,x_ref,name,**kwargs):
        attack = ZooAttack(classifier=self.bbClassifier,confidence=kwargs['confidence'],targeted=self.targeted,learning_rate=kwargs['lr'],
                           max_iter=kwargs['iter_max'],binary_search_steps=kwargs["bs_step"],initial_const=kwargs["initial_const"],
                           abort_early=kwargs["early_stop"],use_resize=kwargs["use_resize"],use_importance=kwargs["use_importance"],
                           nb_parallel=kwargs["nb_patallel"],batch_size=self.batchsize,variable_h=kwargs["variable_h"],verbose=self.verbose)
        x_adv = None
        y_ref = self.APICall.predict(x = x_ref)  
        self.APICall.resetCounts()

        x_adv = attack.generate(x=x_ref)
        y_adv=attack.estimator.predict(x=x_adv,batch_size=self.batchsize) 
        self.APICall.decrementCounts()  

        self.x_list[name] = [x_ref.squeeze(),x_adv.squeeze()]  
        self.y_list[name] = [np.argmax(y_ref),np.argmax(y_adv)]
        self.querycount[name] = [0,self.APICall.getCounts()]



class Square(EvasionAttack):
    pass
