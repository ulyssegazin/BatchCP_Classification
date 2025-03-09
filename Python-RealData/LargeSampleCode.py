import GeneralBatchCode

def LocalPowerAndCoverage(alpha,pVal,I,Ncal,Function,GeneralQuantileFunction,nClass,ytest):
    ntest=len(I)
    y=np.zeros(ntest)
    NbCompt=0
    nBoucle=nClass**ntest
    QuantileFunction=GeneralQuantileFunction(alpha,nClass,Ncal,ntest,N=10**3)
    nPredict=0
    Coverage=0
    yTarget=ytest[I]
    #print('Target'+str(yTarget))
    while NbCompt<nBoucle:
       
        if IndicatorPredictionSet(y,pVal[I],Function,QuantileFunction,nClass):
            nPredict+=1
            if not Coverage:
                #print(y)
                if (y==yTarget).all():
                    Coverage=1
                    
        y=CompteurYclass(y,nClass)
        
        NbCompt+=1
    
    
    Power=1-(nPredict-Coverage)/(nClass**ntest-1)
    return(Coverage,Power,nPredict)

def PowerAndCoverage3MethodsRandomized(Alpha,LenI,WantLencal,X,y,pValuesFunc,Classifieur,Function,GeneralQuantileFunction,nClass,N=10**3):
    nAlpha=len(Alpha)
    
    nMethods=len(Function)
    
    TotCoverage=np.zeros((nMethods,nAlpha,N))
    
    TotPower=np.zeros((nMethods,nAlpha,N))
    
    TotnPredict=np.zeros((nMethods,nAlpha,N))
    
    
    IUniversal=np.arange(LenI).astype(int)

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
                TotCoverage[k][i][j],TotPower[k][i][j],TotnPredict[k][i][j]=LocalPowerAndCoverage(alpha,pVal,IUniversal,Ncal,Function[k],GeneralQuantileFunction,nClass,ytestCurr)
            
            
        #MeanCoverage[i]=np.sum(TotCoverage[i])/N
        
        #MeanPower[i]=np.sum(TotPower[i])/N
        
        #MeannPredict[i]=np.sum(TotnPredict)/N
    
    return(TotCoverage,TotPower,TotnPredict)



def PowerAndCoverage_Full_v_Class(Alpha,LenI,WantLencal,PropClassCal,PropClassTest,X,y,ClassifiedIndices,Classifieur,Function,GeneralQuantileFunction,nClass,N=10**3):
    nAlpha=len(Alpha)
    
    nMethods=len(Function)
    
    TotCoverage=np.zeros((2,nMethods,nAlpha,N))
    
    TotPower=np.zeros((2,nMethods,nAlpha,N))
    
    TotnPredict=np.zeros((2,nMethods,nAlpha,N))
    
    nCalTab=(np.floor(PropClassCal*WantLencal)).astype(int)
    nCalTot=int(np.sum(nCalTab))
    nTestTab=(np.floor(PropClassTest*LenI)).astype(int)
    nTestTot=int(np.sum(nTestTab))
    IUniversal=np.arange(nTestTot).astype(int)
    print(nCalTab)

    print(nTestTab)
    for (i,alpha) in  enumerate(Alpha):
        print(i)
        for j in range(N):
            XcalCurr=np.zeros((nCalTot,np.size(X[0])))
            ycalCurr=np.zeros(nCalTot).astype(int)
            XtestCurr=np.zeros((nTestTot,np.size(X[0])))
            ytestCurr=np.zeros(nTestTot).astype(int)
            CurrIndiceCal=0
            CurrIndiceTest=0
            for numberClass in range(nClass):
                #print('Class'+str(numberClass))
                TotChoice=np.random.choice(len(ClassifiedIndices[numberClass]),size=nTestTab[numberClass]+nCalTab[numberClass],replace=False)
                I=TotChoice[:nTestTab[numberClass]]
                Jcal=TotChoice[nTestTab[numberClass]:]
                XcalCurr[CurrIndiceCal:CurrIndiceCal+nCalTab[numberClass]]=X[Jcal]
                ycalCurr[CurrIndiceCal:CurrIndiceCal+nCalTab[numberClass]]=int(numberClass)
                XtestCurr[CurrIndiceTest:CurrIndiceTest+nTestTab[numberClass]]=X[I]
                ytestCurr[CurrIndiceTest:CurrIndiceTest+nCalTab[numberClass]]=int(numberClass)
                CurrIndiceCal+=nCalTab[numberClass]
                CurrIndiceTest+=nTestTab[numberClass]
            
            (pValClass,NcalClass)=ClassCalibratedpvalues(Classifieur,nClass,XcalCurr,ycalCurr,nCalTot,XtestCurr,nTestTot)
            (pValFull,NcalFull)=FullCalibratedpvalues(Classifieur,nClass,XcalCurr,ycalCurr,nCalTot,XtestCurr,nTestTot)
            for k in range (nMethods):    
                TotCoverage[0][k][i][j],TotPower[0][k][i][j],TotnPredict[0][k][i][j]=LocalPowerAndCoverage(alpha,pValClass,IUniversal,NcalClass,Function[k],GeneralQuantileFunction,nClass,ytestCurr)
                TotCoverage[1][k][i][j],TotPower[1][k][i][j],TotnPredict[1][k][i][j]=LocalPowerAndCoverage(alpha,pValFull,IUniversal,NcalFull,Function[k],GeneralQuantileFunction,nClass,ytestCurr)
            
            
        #MeanCoverage[i]=np.sum(TotCoverage[i])/N
        
        #MeanPower[i]=np.sum(TotPower[i])/N
        
        #MeannPredict[i]=np.sum(TotnPredict)/N
    
    return(TotCoverage,TotPower,TotnPredict)
