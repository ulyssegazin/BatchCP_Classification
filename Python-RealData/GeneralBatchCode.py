# Conformal $p$-values

### Computation of conformal p-values with Classifieur, (Xcal,ycal) calibration set and xtest a set of test point 

def FullCalibratedpvalues(Classifieur,nclass,Xcal,ycal,ncal,xtest,ntest):
    Probacal=1-Classifieur.predict_proba(Xcal)
    Probatest=1-Classifieur.predict_proba(xtest)
    
    ScoreCal=Probacal[np.arange((ncal)),ycal]
    
    pValues=np.zeros((ntest,nclass))
    for i in range (ntest):
        for j in range (nclass):
            pValues[i][j]=1 + np.sum(ScoreCal >= Probatest[i][j])
    
    return (pValues/(ncal+1),ncal)


def ClassCalibratedpvalues(Classifieur,nclass,Xcal,ycal,ncal,xtest,ntest):
    Probacal=1-Classifieur.predict_proba(Xcal)
    Probatest=1-Classifieur.predict_proba(xtest)
    
    ScoreCal=Probacal[np.arange((ncal)),ycal]
    ncalClass=np.zeros(nclass)
    
    pValues=np.zeros((ntest,nclass))
    for j in range (nclass):    
        ScoreCalClass=ScoreCal[np.nonzero(ycal==j)]
        ncalClass[j]=len(ScoreCalClass)
        
        for i in range (ntest):
            pValues[i][j]=(1 + np.sum(ScoreCalClass >= Probatest[i][j]))/(ncalClass[j]+1)
    
    return (pValues,ncalClass.astype(int))

#Simulation of  conformal $p$-values
def Distrib_EmpiricalPvalues(n):
    U=stat.dirichlet.rvs(alpha=np.ones(n+1))[0]
    Val=(np.arange(n+1)+1)/(n+1)
    

    DistribCond=stat.rv_discrete(values=(Val,U))
    return DistribCond

def Simu_FullConformallPvalues(n,m,N=1):
    
    T=np.zeros((N,m))
    for i in range(N):
        DistribCond=Distrib_EmpiricalPvalues(n)
        T[i]=DistribCond.rvs(size=m)

    return T

def Simu_ClassConformallPvalues(nTab,my,N=1):
    m=int(np.sum(my))
    nClass=len(my)
    T=np.zeros((N,m))
    for i in range(N):
        Position=0
        for j in range (nClass):
            
            DistribCond=Distrib_EmpiricalPvalues(nTab[j])
            T[i][Position:Position+my[j]]=DistribCond.rvs(size=my[j])
            Position+=my[j]
            
            
    return T


# Combination Function


def BonfFunction(pVal):
    m=len(pVal)
    return np.min(pVal)*m



def SimesFunction(pVal):
    m=len(pVal)
    pValOrd=np.sort(pVal)
    mm=(np.arange(m)+1)/m
    SimesFamily=pValOrd/mm
    return np.min(SimesFamily)

def AdaptiveSimesFunctionGenerator(lamb):
    def AdaptiveSimesFunction(pVal):
        m=len(pVal)
        mStorey=(1+np.sum(pVal>lamb))/(1*(1-lamb))
        
        pValOrd=np.sort(pVal)
        mm=(np.arange(m)+1)/mStorey
        SimesAdaptFamily=pValOrd/mm
        return np.min(SimesAdaptFamily)
    return AdaptiveSimesFunction
    
SimesStorey1_2=AdaptiveSimesFunctionGenerator(1/2)
SimesStorey5_100=AdaptiveSimesFunctionGenerator(0.05) 



def QuantileSimesFunctionGenerator(lamb):
    def QuantileSimesFunction(pVal):
        m=len(pVal)
        Quant=int(np.ceil(m*lamb))
        pValOrd=np.sort(pVal)
        mStorey=(1+m-Quant)/(1*(1-pValOrd[Quant-1]))
        
        
        mm=(np.arange(m)+1)/mStorey
        SimesAdaptFamily=pValOrd/mm
        return np.min(SimesAdaptFamily)
    return QuantileSimesFunction




def FisherFunction(pVal):
    m=len(pVal)
    FisherStat=-2*np.sum(np.log(pVal))
    return stat.chi2.sf(FisherStat,df=2*m)


# Threshold Function

#### Reject if $F(p)>alpha #####
def AlphaLevel(alpha,nclass,ncalClass,ntest,N=10**3):
    return lambda my : alpha


#### Approximate the alpha quantile of F(p) when p is a family of fully calibrated p-values with good labels####
def EmpiricalQuantileFull(alpha,Function,nclass,ncal,N=10**3):
    Ptab=Simu_FullConformallPvalues(ncal,ntest,N)
    FunctionNullTab=np.zeros(N)
    for i in range (N):
        FunctionNullTab[i]=Function(Ptab[i])
    FunctionNullOrd=np.sort(FunctionNullTab)
    return lambda my : FunctionNullTab[int(np.ceil((N+1)*(alpha)))]



#### Approximate the alpha quantile of F(p) when p is a family of class calibrated p-values with good labels ####
def EmpiricalQuantileClass(alpha,Function,nclass,ncal,N=10**3):
    def ClassQuantileSimesFunction(my):
        Ptab=Simu_ClassConformallPvalues(ncal,my,N)
        FunctionNullTab=np.zeros(N)
        for i in range (N):
            FunctionNullTab[i]=Function(Ptab[i])
        FunctionNullOrd=np.sort(FunctionNullTab)
        return FunctionNullOrd[int(np.ceil((N+1)*(alpha)))]
    return ClassQuantileSimesFunction


# Creation of the batch prediction region


#### Compute if y is in the batch prediction region ####
def IndicatorPredictionSet(y,pVal,Function,QuantileFunction,nclass):
    my=np.zeros(nclass)
    for i in range(nclass):
        my[i]=(y==i).sum()
    return Function(pVal[np.arange(len(pVal)),y.astype(int)])>QuantileFunction(my.astype(int))



#### Seek for all the possible label with the same size as y and with nClass possible Class 
def CompteurYclass(y,nClass):
    Current=0
    Retenue=True
    ylen=len(y)
    while Retenue and Current<ylen:
        if y[Current]==nClass-1:
            y[Current]=0
            Current+=1
        else:
            y[Current]+=1
            Retenue=False
    return y

#### Compute the batch prediction region ####
def PredictionSet(alpha,pVal,I,Ncal,Function,GeneralQuantileFunction,nClass):
    ntest=len(I)
    YPredict=np.array(-np.ones(ntest))
    y=np.zeros(ntest)
    NbCompt=0
    nBoucle=nClass**ntest
    QuantileFunction=GeneralQuantileFunction(alpha,nClass,Ncal,ntest,N=10**3)
    nPredict=0
    while NbCompt<nBoucle:
       
        if IndicatorPredictionSet(y,pVal[I],Function,QuantileFunction,nClass):
            YPredict=np.vstack([YPredict,[y]])
            #print(y)
            nPredict+=1
            
            
            
        y=CompteurYclass(y,nClass)
        
        NbCompt+=1
    if nPredict==0:
        return[]
    else:
        return(YPredict[1:])
    
# Computation of bounds


#### Given a batch prediction set, return four array with respectively the lower bounds for each class,
#### if we have updated the lower bound, the upper bounds for each class and if updated the upper bound 

def PostHocBounds(YPredict,nClass,I):
    ntest=len(I)
    nSelec=len(YPredict)
    LowBounds=np.ones(nClass)*ntest
    UpBounds=np.ones(nClass)*0
    NotEmptyLow=np.zeros(nClass)
    NotEmptyUp=np.zeros(nClass)
    for y in YPredict:
        for i in range(nClass):
            Nbcurr=(y==i).sum()
            if Nbcurr<LowBounds[i]:
                LowBounds[i]=Nbcurr
                NotEmptyLow[i]=True
            if Nbcurr>UpBounds[i]:
                UpBounds[i]=Nbcurr
                NotEmptyUp[i]=True
                
    for i in range(nClass):
        if not NotEmptyLow[i]:
            
            LowBounds[i]=0
        if not NotEmptyLow[i]:
            
            UpBounds[i]=m
    return (LowBounds,NotEmptyLow,UpBounds,NotEmptyUp)


#### Same function but with the exact label ####
def PostHocBoundsLabels(YPredict,Label,I):
    ntest=len(I)
    nClass=len(Label)
    nSelec=len(YPredict)
    LowBounds=np.ones(nClass)*ntest
    UpBounds=np.ones(nClass)*0
    NotEmptyLow=np.zeros(nClass)
    NotEmptyUp=np.zeros(nClass)
    for y in YPredict:
        for i in range(nClass):
            labelCurr=Label[i]
            Nbcurr=(y==labelCurr).sum()
            if Nbcurr<LowBounds[i]:
                LowBounds[i]=Nbcurr
                NotEmptyLow[i]=True
            if Nbcurr>UpBounds[i]:
                UpBounds[i]=Nbcurr
                NotEmptyUp[i]=True
                
    for i in range(nClass):
        if not NotEmptyLow[i]:
            print("Vide Low")
            LowBounds[i]=0
        if not NotEmptyLow[i]:
            print("Vide Up")
            UpBounds[i]=m
    return (LowBounds,NotEmptyLow,UpBounds,NotEmptyUp)

def Storeym0FunctionGenerator(lamb):
    def StoreyFunction(pVal):
        m=len(pVal)
        mStorey=min((1+np.sum(pVal>lamb))/(1*(1-lamb)),m)
        

        return mStorey
    return StoreyFunction
 
    
def Bonfm0(pVal):
    m=len(pVal)
    return m    



#### Implementation of the shortcuts
def ShortcutBounds(pVal,alpha,m0Func,m,nClass):
    pValTrans=pVal.T
    TabH=np.zeros((nClass,m+1))
    Low=m*np.ones(nClass)
    Up=np.zeros(nClass)
    for k in range (nClass):
        
        pValTrans[k]=-np.sort(-pValTrans[k])

        bValues=np.max(np.delete(pValTrans, k, axis = 0),axis=0)
        bValues=-np.sort(-bValues)
        
        
        
        SeeUp=False
        for i in range(m+1):
            qValues=np.zeros(m)
            qValues[:m-i]=pValTrans[k][:m-i]
            qValues[m-i:]=bValues[:i]
            
            m0=m0Func(qValues)
            mm=(np.arange(m)+1)/m0
            TabH[k][m-i]=np.min(qValues/mm)
            
            if TabH[k][m-i]>alpha:
                Low[k]=m-i
                if not SeeUp:
                    Up[k]=m-i
                    SeeUp=True
        

            
    return(Low,Up)


# Batch Scores with the LRT

def LRTStatistics(Classifieur,X,y):
    ScoreOpposite=Classifieur.predict_proba(X)
    TabLRT=np.max(ScoreOpposite,axis=1)/ScoreOpposite[np.arange(len(X)),y]
    return np.prod(TabLRT)

def LRTStatisticsScores(ScoreOpposite,Indices,yIndices):
    InterestingScores=ScoreOpposite[Indices]
    TabLRT=np.max(InterestingScores,axis=1)/InterestingScores[np.arange(len(Indices)).astype(int),yIndices.astype(int)]
    return np.prod(TabLRT)

def LRTIndicator(alpha,Classifieur,Bperm,xtest,ytest,Xtot,ycal,ntest,ntot):
    
    ytot=np.concatenate((ycal,ytest))

    TabLRTStat=np.zeros(Bperm)
    LRTStatTest=LRTStatistics(Classifieur,xtest,ytest)
    for i in range(Bperm):
        SelecIndices=np.random.choice(ntot,size=ntest,replace=False).astype(int)
        TabLRTStat[i]=LRTStatistics(Classifieur,Xtot[SelecIndices],ytot[SelecIndices])
    
    pvalue=(1+np.sum(TabLRTStat>=LRTStatTest))/(Bperm+1)
    return (pvalue>alpha)


#ScoresOpposite ncal first score for the X in calibration, ntest last for the X in test

def LRTIndicatorScores(alpha,Bperm,ytest,ycal,ScoreOpposite,ntest,ncal,ntot):
    
    ytot=np.concatenate((ycal,ytest))
    
    IndicesTest=ncal+np.arange(ntest)
    TabLRTStat=np.zeros(Bperm)
    LRTStatTest=LRTStatisticsScores(ScoreOpposite,IndicesTest,ytest)
    for i in range(Bperm):
        SelecIndices=np.random.choice(ntot,size=ntest,replace=False).astype(int)
        TabLRTStat[i]=LRTStatisticsScores(ScoreOpposite,SelecIndices,ytot[SelecIndices])
    
    pvalue=(1+np.sum(TabLRTStat>=LRTStatTest))/(Bperm+1)
    return [pvalue>alpha,pvalue]


def LRTBatchPrediction(alpha,Bperm,Classifieur,Xcal,ycal,xtest,nClass):
    
    ntest=len(xtest)
    ncal=len(ycal)
    Xtot=np.concatenate((Xcal,xtest))
    ntot=len(Xtot)
    ScoreOpposite=Classifieur.predict_proba(Xtot)
    
    YPredict=np.array(-np.ones(ntest))
    y=np.zeros(ntest)
    NbCompt=0
    nBoucle=nClass**ntest
    nPredict=0
    
    
    while NbCompt<nBoucle:    
        if LRTIndicatorScores(alpha,Bperm,y.astype(int),ycal,ScoreOpposite,ntest,ncal,ntot)[0]:
            YPredict=np.vstack([YPredict,[y]])
            #print(y)
            nPredict+=1



        y=CompteurYclass(y,nClass)

        NbCompt+=1
    if nPredict==0:
        return[]
    else:
        return(YPredict[1:])

def LocalPowerAndCoverageLRT(alpha,Bperm,ScoreOpposite,ytot,IndicesCal,IndicesTest,nClass):
    
    
    AllIndices=np.concatenate((IndicesCal,IndicesTest))
    
    ScoreOppositeSelect=ScoreOpposite[AllIndices]
    ntest=len(IndicesTest)
    ncal=len(IndicesCal)
    ntot=ncal+ntest
    y=np.zeros(ntest)
    NbCompt=0
    nBoucle=nClass**ntest

    nPredict=0
    Coverage=0
    yTarget=ytot[IndicesTest]
    #print('Target'+str(yTarget))
    while NbCompt<nBoucle:
       
        if LRTIndicatorScores(alpha,Bperm,y.astype(int),ytot[IndicesCal],ScoreOppositeSelect,ntest,ncal,ntot)[0]:
            nPredict+=1
            if not Coverage:
                #print(y)
                if (y==yTarget).all():
                    Coverage=1
                    
        y=CompteurYclass(y,nClass)
        
        NbCompt+=1
    
    
    Power=1-(nPredict-Coverage)/(nClass**ntest-1)
    return(Coverage,Power,nPredict)





def PowerAndCoverage3MethodsRandomizedWithLRT(Alpha,Bperm,LenI,WantLencal,X,y,pValuesFunc,Classifieur,Function,GeneralQuantileFunction,nClass,N=10**3):
    nAlpha=len(Alpha)
    
    nMethods=len(Function)
    
    TotCoverage=np.zeros((nMethods+1,nAlpha,N))
    
    TotPower=np.zeros((nMethods+1,nAlpha,N))
    
    TotnPredict=np.zeros((nMethods+1,nAlpha,N))
    
    Times=np.zeros((nMethods+1,nAlpha,N))
    
    IUniversal=np.arange(LenI).astype(int)
    ScoreOpposite=Classifieur.predict_proba(X)
    for (i,alpha) in  enumerate(Alpha):
        print(i)
        for j in range(N):
            
            TotChoice=np.random.choice(len(y),size=LenI+WantLencal,replace=False)
            I=TotChoice[:LenI]
            Jcal=TotChoice[LenI:]
            XcalCurr=X[Jcal]
            ycalCurr=y[Jcal]
            XtestCurr=X[I]
            ytestCurr=y[I]

            (pVal,Ncal)=pValuesFunc(Classifieur,nClass,XcalCurr,ycalCurr,WantLencal,XtestCurr,LenI)
    
            for k in range (nMethods):
                BeginTime=perf_counter()
                TotCoverage[k][i][j],TotPower[k][i][j],TotnPredict[k][i][j]=LocalPowerAndCoverage(alpha,pVal,IUniversal,Ncal,Function[k],GeneralQuantileFunction,nClass,ytestCurr)
                EndTime=perf_counter()
                Times[k][i][j]=EndTime-BeginTime
            
            BeginTime=perf_counter()
            TotCoverage[nMethods][i][j],TotPower[nMethods][i][j],TotnPredict[nMethods][i][j]=LocalPowerAndCoverageLRT(alpha,Bperm,ScoreOpposite,y,Jcal,I,nClass)
            EndTime=perf_counter()
            Times[nMethods][i][j]=EndTime-BeginTime
            
        #MeanCoverage[i]=np.sum(TotCoverage[i])/N
        
        #MeanPower[i]=np.sum(TotPower[i])/N
        
        #MeannPredict[i]=np.sum(TotnPredict)/N
    
    return(TotCoverage,TotPower,TotnPredict,Times)