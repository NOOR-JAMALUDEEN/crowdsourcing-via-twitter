# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 14:26:57 2018

@author: HP
"""



import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from random import randrange
import csv
from pathlib import Path

#=============================Topic Independent Simulator ================================================
def simulateLabels(groundTruth,tweets_topics,topics,numberAnnotators,meanAccuarcy,SdAccuracy,meanLikelihood,nb_labels):
  x=[]
  likelihood=[]
  numberTopics=len(tweets_topics[0])
  numberTweets=len(tweets_topics)
  print(numberTweets)
  annt_responses=np.full((numberAnnotators,numberTweets),0)
  annt_topics=np.full((numberAnnotators,nb_topics),1.0)
  annt_topics_float=np.full((numberAnnotators,nb_topics),1.0)
  for m in range(0,numberAnnotators):
   done=False
   while(done==False):
    val=np.random.normal(loc=meanAccuarcy,scale=SdAccuracy ,size=1)#the accurcy for each annotater
    val=val/100
    
    if val>0 and val<1:
        x.append(val)
        done=True
        
  for m in range(0,numberAnnotators):
   done=False
   while(done==False):
    val1=np.random.exponential(scale=meanLikelihood,size=1)
    val1=val1/100
    if val1>0.0 and val1<1.0:
       likelihood.append(val1)
       ##likelihood.append(0.005)
       done=True
       
  annt_counter=[]
  for m in range(0,numberAnnotators):
      counter=0
      for i in range(0,numberTweets):
        correct=np.random.binomial(1,x[m],1)
        annotate=np.random.binomial(1,likelihood[m],1)
        if (annotate[0]!=0.0):
         counter=counter+1
         if correct[0]==1:   
          annt_responses[m,i]=groundTruth[i]
          for c in range(0,numberTopics):
                if tweets_topics[i,c]!=0.0:
                    annt_topics[m,c]=annt_topics[m,c]+1
                    annt_topics_float[m,c]=round(annt_topics[m,c]+(tweets_topics[i,c]),4)
                    
         else:
           annt_responses[m,i]=randrange(1,nb_labels+1,1)  
           if annt_responses[m,i]==groundTruth[i]:
               for c in range(0,nb_topics):
                if tweets_topics[i,c]!=0.0:
                    annt_topics[m,c]=annt_topics[m,c]+1
                    annt_topics_float[m,c]=round(annt_topics[m,c]+(tweets_topics[i,c]),4)
           
      
  return (annt_topics,annt_topics_float,annt_responses)

###======================================Topic dependent Simulator ==================================================
def simulateLabelsTopicsDependent(groundTruth,tweets_oneTopic,topics,numberAnnotators,reliablePercentage,SdAccuracy,meanLikelihood,nb_labels):
  likelihood=[]
  numberTopics=len(tweets_oneTopic[0])
  numberTweets=len(tweets_oneTopic)
  ##print(numberAnnotators)
  annt_responses=np.full((numberAnnotators,numberTweets),0)
  annt_oneTopic=np.full((numberAnnotators,numberTopics),1.0)
  annt_topics=np.full((numberAnnotators,numberTopics),1.0)
  annt_topics_float=np.full((numberAnnotators,numberTopics),1.0)
  
  for m in range(0,numberAnnotators):
   done=False
   while(done==False):
    val1=np.random.exponential(scale=meanLikelihood,size=1)
    val1=val1/100
    if val1>0.0 and val1<1.0:
       likelihood.append(val1)
       ##likelihood.append(0.5)
       done=True
  annt_counter=[]
  for m in range(0,numberAnnotators):
      counter=0
      ##print("annotator",m)
      for i in range(0,numberTopics):
        is_reliable=np.random.binomial(1,reliablePercentage,1)
        if is_reliable:
         ##print("reliable",i)   
         done=False
         while(done==False):
          val=np.random.normal(80,scale=10,size=1)#topic reliability
          val=val/100
          if val>0 and val<1:
            annt_oneTopic[m][i]=val
            done=True
        else:
         ##print("unreliable",i)
         done2=False
         while(done2==False):
          val=np.random.normal(10,scale=10,size=1)#topic reliability
          val=val/100
          if val>0 and val<1:
            annt_oneTopic[m][i]=val
            done2=True   
      for tweetsCounter in range(0,numberTweets):
        correctProbability=0
        correctProbability2=0
        normalizeFactor=0
        annotate=np.random.binomial(1,likelihood[m],1)
        ##annotate=np.random.binomial(1,0.9,1) 
        if (annotate[0]!=0.0):
         for topicCounter in range(0,nb_topics):
          if tweets_oneTopic[tweetsCounter][topicCounter]!=0:
           ##if annt_oneTopic[m][topicCounter]>=0.5:
            ##correctProbability2=correctProbability2+(tweets_oneTopic[tweetsCounter][topicCounter])
           correctProbability=correctProbability+(tweets_oneTopic[tweetsCounter][topicCounter]*annt_oneTopic[m][topicCounter])
           normalizeFactor=normalizeFactor+tweets_oneTopic[tweetsCounter][topicCounter]
         if normalizeFactor!=0.0:   
          correctProbability=correctProbability/normalizeFactor
         else:
          correctProbability=0.0
         correct=np.random.binomial(1,correctProbability,1)
         ##correct=bernoulli.rvs(correctProbability, size=1)
         counter=counter+1
         if correct[0]==1:   
          annt_responses[m,tweetsCounter]=groundTruth[tweetsCounter]
          for c in range(0,numberTopics):
                if tweets_oneTopic[tweetsCounter,c]!=0.0:
                    ##print("correct topic",c,"value",round(annt_oneTopic[m,c],4),"tweet",tweetsCounter,"with value",round(tweets_oneTopic[tweetsCounter,c],4),round(correctProbability,4),"pro2",round(correctProbability2/normalizeFactor,4))
                    annt_topics[m,c]=annt_topics[m,c]+1
                    annt_topics_float[m,c]=round(annt_topics[m,c]+(tweets_oneTopic[tweetsCounter,c]),4)
                    
         else:
           annt_responses[m,tweetsCounter]=randrange(1,nb_labels+1,1)  
           if annt_responses[m,tweetsCounter]==groundTruth[tweetsCounter]:
               for c in range(0,numberTopics):
                if tweets_oneTopic[tweetsCounter,c]!=0.0:
                    annt_topics[m,c]=annt_topics[m,c]+1
                    annt_topics_float[m,c]=round(annt_topics[m,c]+(tweets_oneTopic[tweetsCounter,c]),4)
           '''else:
               for c in range(0,numberTopics):
                if tweets_topics[i,c]!=0.0:
                    #annt_topics[m,c]=round(annt_topics[m,c]-(tweets_topics[i,c]/topics[c]),4)
                    annt_topics[m,c]=round(annt_topics[m,c]-(tweets_topics[i,c]),4)
             '''
     
  return (annt_topics,annt_topics_float,annt_responses)

###========================================================================================



def calculateReliability(groundTruth,annt_responses,nb_topics,numberTweets,tweets_topics_local):
  annt_topics=np.full((len(annt_responses),nb_topics),1.0)
  annt_topics_float=np.full((len(annt_responses),nb_topics),1.0)
  for m in range(0,len(annt_responses)):
      for i in range(0,numberTweets):
           if annt_responses[m,i]==groundTruth[i]:
               for c in range(0,nb_topics):
                if tweets_topics_local[i,c]!=0.0:
                    annt_topics[m,c]=annt_topics[m,c]+1
                    annt_topics_float[m,c]=round(annt_topics[m,c]+(tweets_topics[i,c]),4)
      
  return (annt_topics,annt_topics_float)


#=============== Kappa inter-agreement=============================
def kappa_aggreement(annt_responses_local,nb_labels):
    kappa_agree=[]
    nb_annotators=len(annt_responses_local)
    nb_tweets=len(annt_responses_local[0])
    
    for i in range(0,nb_annotators):
     norm=0
     kappa=0.0
     for j in range(0,nb_annotators):
        common=False
        confusion=np.full((nb_labels,nb_labels),0)
        for z in range(0,nb_tweets):
            for l in range(1,nb_labels+1):
               
               if annt_responses_local[i][z]!=0 and annt_responses_local[j][z]!=0:
                   common=True 
                   confusion[(annt_responses_local[i][z])-1][(annt_responses_local[j][z])-1]=confusion[(annt_responses_local[i][z])-1][(annt_responses_local[j][z])-1]+1
        if common==True:
            norm=norm+1
        total=confusion.sum()
        pra=0.0
        if total!=0.0:
         pra=np.trace(confusion)/total
        pre=0.0
        cols=confusion.sum(axis=0)
        rows=confusion.sum(axis=1)
        for d in range(0,nb_labels):
          if total!=0.0:
            pre=pre+(cols[d]*rows[d])/total
        if total!=0.0: 
         pre=pre/total
        if pre!=1: 
         kappa=kappa+((pra-pre)/(1.0-pre))
     if (norm!=0):    
      kappa_agree.append(kappa/(norm))
     else:
      kappa_agree.append(0.0)  
    ##print('kappa_Agree',kappa_agree)
    return kappa_agree
 
#===============End Kappa inter- agreement========================
#============Kappa with topics============================
def kappaInteragreemtWithTopics(annt_responses_local,nb_labels,tweets_topics_local,groundTruth_local):
    kappa_agree=kappa_aggreement(annt_responses_local,nb_labels)
    nb_annotators=len(annt_responses_local)
    nb_tweets=len(annt_responses_local[0])
    nb_topics=len(tweets_topics_local[0])
    annt_tpc_kappa=np.full((nb_annotators,nb_topics),1.0)
    Kappa_trueLabels=[]
    onlylabel=0.0
    for i in range(0,nb_tweets):
     highsim=0.0
     truelabel=0
     for label in range(1,nb_labels+1):
         sim=0.0
         for j in range(0,nb_annotators):
             if annt_responses_local[j][i]==label:
                 onlylabel=label
                 for c in range(0,nb_topics):
                  if tweets_topics_local[i,c]!=0.0:
                   ##sim=sim+(kappa_agree[j]*annt_tpc_kappa[j,c])
                   sim=sim+(kappa_agree[j])
         if highsim<sim:
             truelabel=label
             highsim=sim
     if truelabel!=0.0:        
      Kappa_trueLabels.append(truelabel)
     else:
      truelabel=onlylabel   
      Kappa_trueLabels.append(onlylabel)   
     for j in range(0,nb_annotators):
      if (annt_responses_local[j][i]==truelabel) :
              highestTopic=0
              highestTopicValue=tweets_topics_local[i,0]
              for c in range(0,nb_topics):
                  if tweets_topics_local[i,c]!=0.0:
                      annt_tpc_kappa[j,c]=annt_tpc_kappa[j,c]+1
                      '''if tweets_topics[i,c]>highestTopicValue:
                       highestTopic=tweets_topics[i,c]
                       highestTopic=c'''
                ##annt_tpc_kappa[j,c]=round(annt_tpc_kappa[j,c]+(tweets_topics[i,c]),4)
              ##annt_tpc_kappa[j,highestTopic]=annt_tpc_kappa[j,highestTopic]+1
     ''' 
     else:
             for c in range(0,nb_topics):
                  if tweets_topics[i,c]!=0.0 and annt_responses[j][i]!=0 :
                    annt_tpc_kappa[j,c]=round(annt_tpc_kappa[j,c]-(tweets_topics[i,c]),4)
                    ##annt_tpc_kappa[j,c]=annt_tpc_kappa[j,c]-1
    ##print('kappTopics',Kappa_trueLabels)
    ##print(annt_tpc_kappa)'''
    return(Kappa_trueLabels,annt_tpc_kappa)
#==============End Kappa with topics=================================
#============Kappa without topics============================
def kappaInterAgreementWithoutTopics(annt_responses,nb_labels):
    Kappa_trueLabelsWithoutTopics=[]
    kappa_agree=kappa_aggreement(annt_responses,nb_labels)
    nb_tweets=len(annt_responses[0])
    nb_annotators=len(annt_responses)
    for i in range(0,nb_tweets):
     highsim=0.0
     truelabel=0
     for label in range(1,nb_labels+1):
         sim=0.0
         for j in range(0,nb_annotators):
             if annt_responses[j][i]==label:
                 for c in range(0,nb_topics):
                  if tweets_topics[i,c]!=0.0:
                   sim=sim+(kappa_agree[j])
         if highsim<sim:
             truelabel=label
             highsim=sim
     Kappa_trueLabelsWithoutTopics.append(truelabel)
    return Kappa_trueLabelsWithoutTopics
#==============End Kappa without topics=================================

#===========Majority Voting===============================
def majorityVoting(annt_responses_local,nb_labels,nb_topics,tweets_topics_local):
    majority_voting=[]
    annt_mv_tpc=np.full((len(annt_responses_local),nb_topics),1.0)
    nb_tweets=len(annt_responses_local[0])
    nb_annotators=len(annt_responses_local)
    for j in range(0,nb_tweets):
            high=0
            s=0
            majority=0
            for x in range(1,nb_labels+1):
             s=0
             for i in range(0,nb_annotators):
                if annt_responses_local[i][j]==x:
                    s=s+1
             if s>high:
              high=s
              majority=x
            majority_voting.append(majority)
            for l in range(0,nb_annotators):
             for c in range(0,nb_topics):
                if (annt_responses_local[l][j]==majority) and (tweets_topics_local[j][c]!=0) :
                    annt_mv_tpc[l][c]=annt_mv_tpc[l][c]+1
                    ##annt_mv_tpc[l][c]=annt_mv_tpc[l][c]+tweets_topics[j][c]
    return (majority_voting,annt_mv_tpc)

def mvWithTopics(annt_responses_local,nb_topics,nb_labels,groundTruth_local,tweets_topics_local):
    nb_tweets=len(annt_responses_local[0])
    trueLabels=[]
    nb_annotators=len(annt_responses_local)
    annt_tpc=np.full((len(annt_responses_local),nb_topics),1.0)
    annt_tpc_test=np.full((len(annt_responses_local),nb_topics),1.0)
    for i in range(0,nb_tweets):
     highsim=0.0
     truelabel=0
     counter=0
     for nA in range(0,len(annt_responses_local)):
         if(annt_responses_local[nA,i]!=0):
          counter=counter+1
     for label in range(1,nb_labels+1):
         sim=0.0
         accm=0
         for j in range(0,nb_annotators):
             if annt_responses_local[j][i]==label:
                 
                 for c in range(0,nb_topics):
                  if tweets_topics_local[i,c]!=0.0:
                   accm=accm+(annt_tpc[j,c])
         if counter!=0:
          sim=accm/counter
         
         if highsim<sim:
             truelabel=label
             highsim=sim
     trueLabels.append(truelabel)
     
     for k in range(0,nb_annotators):
      if (annt_responses_local[k][i]==truelabel) :
              highestTopic=0
              highestTopicValue=tweets_topics_local[i,0]
              for c in range(0,nb_topics):
                  if tweets_topics_local[i,c]!=0.0:
                   '''if tweets_topics[i,c]>highestTopicValue:
                       highestTopic=tweets_topics[i,c]
                       highestTopic=c'''
              ##annt_tpc_kappa[j,c]=round(annt_tpc_kappa[j,c]+(tweets_topics[i,c]),4)
              ##annt_tpc[k,highestTopic]=annt_tpc[k,highestTopic]+1
              ##annt_tpc[j,highestTopic]=round(annt_tpc[j,highestTopic]+(tweets_topics[i,highestTopic]),4)
                   annt_tpc[k,c]=annt_tpc[k,c]+1
              ##annt_tpc_test=annt_tpc_test+tweets_topics[i,c]
      '''else:
             for c in range(0,nb_topics):
                  if tweets_topics[i,c]!=0.0 and annt_responses[j][i]!=0:
                    #annt_tpc[j,c]=round(annt_tpc[j,c]-(tweets_topics[i,c]/topics[c]),4)
                    annt_tpc[j,c]=round(annt_tpc[j,c]-(tweets_topics[i,c]),4)
                    ##annt_tpc[j,c]=annt_tpc[j,c]-1'''
    ##return (trueLabels,annt_tpc,annt_tpc_test)
    return (trueLabels,annt_tpc)
 
def accuracy(trueLabels_local,groundTruth_local,text,annt_responses,tweets_topics_local):
    hits=0
    nb_tweets=len(trueLabels_local)
    for i in range(0,nb_tweets):
     if groundTruth_local[i]==trueLabels_local[i]:
        hits=hits+1
    print('accuracy',text,(float(hits)/float(nb_tweets)),'missclasification: ',nb_tweets-hits,'of',nb_tweets)
    
    return (float(hits)/float(nb_tweets))

def accuracyIncremental(trueLabels_local,groundTruth_local,text,annt_responses,tweets_topics_local):
    hits=0
    nb_tweets=len(trueLabels_local)
    for i in range(0,nb_tweets):
     if groundTruth_local[i]==trueLabels_local[i]:
        hits=hits+1
    print('accuracy',text,(float(hits)/float(nb_tweets)),'missclasification: ',nb_tweets-hits,'of',nb_tweets)
    
    return (hits)


def accuracyReliability(realAnntTopics,estAnntTopics):
        print(realAnntTopics)
        print(estAnntTopics)
        nb_annotators=len(realAnntTopics)
        nb_topics=len(realAnntTopics[0])
        error_reliablility=0.0
        for i in range(0,nb_annotators):
            for j in range(0,nb_topics):
                error_reliablility=error_reliablility+((realAnntTopics[i][j]-estAnntTopics[i][j])*(realAnntTopics[i][j]-estAnntTopics[i][j]))
                
        error_reliablility=error_reliablility/(nb_annotators*nb_topics)
        result=math.sqrt(error_reliablility)
        return result

def accumulateReliability(globalAnntTopics,incrementalAnntTopics):
        nb_annotators=len(globalAnntTopics)
        nb_topics=len(globalAnntTopics[0])
        for i in range(0,nb_annotators):
            for j in range(0,nb_topics):
                globalAnntTopics[i][j]=globalAnntTopics[i][j]+incrementalAnntTopics[i][j]
        return globalAnntTopics

#========================accuracy======================================
      
def mainRunWithSparsity(annt_responses_local,annt_topics_local,annt_topics_float,nb_labels,tweets_topics_loc,groundTruth_temp_local):
    nb_topics=len(annt_topics_local[0])
    trueLabels=[]
    annt_tpc=[]
    majority_voting=[]
    annt_mv_tpc=[]
    mv_withTopics=[]
    mv_annt_tpc=[]
    accu=[]
    rel_acc=[]
    
    (trueLabels,annt_tpc)=kappaInteragreemtWithTopics(annt_responses_local,nb_labels,tweets_topics_loc,groundTruth_temp_local)
    rel_acc.append(accuracyReliability(annt_topics_local,annt_tpc))
    accu.append(accuracy(trueLabels,groundTruth_temp_local,'Kappa-agreement with topics',annt_responses_local,tweets_topics_loc))
    print("annt_tpc kappa",annt_tpc)
    print("trueLabels kapppa")
    print(trueLabels)
    
    (mv_withTopics,mv_annt_tpc)=mvWithTopics(annt_responses_local,nb_topics,nb_labels,groundTruth_temp_local,tweets_topics_loc) 
    accu.append(accuracy(mv_withTopics,groundTruth_temp_local,'Majority Voting with Topics',annt_responses_local,tweets_topics_loc))
    rel_acc.append(accuracyReliability(annt_topics_local,mv_annt_tpc))
    print("mv with topics",mv_annt_tpc)
    print("mv_withTopics truelabe")
    print(mv_withTopics)
    
    (majority_voting,annt_mv_tpc)=majorityVoting(annt_responses_local,nb_labels,nb_topics,tweets_topics_loc)
    accu.append(accuracy(majority_voting,groundTruth_temp_local,'Majority Voting',annt_responses_local,tweets_topics_loc))
    rel_acc.append(accuracyReliability(annt_topics_local,annt_mv_tpc))
    
    print("majority voting",annt_mv_tpc)
    print("majority_voting")
    print(majority_voting)
    
    ##print(mv_annt_tpc)
    return (accu,rel_acc)
##==========================================================No Ranking======================================================
    
def mainRunWithSparsityNoranking(annt_responses_local,annt_topics_local,annt_topics_float,nb_labels,tweets_topics_loc,groundTruth_temp_local):
    nb_topics=len(annt_topics_local[0])
    trueLabels=[]
    annt_tpc=[]
    majority_voting=[]
    annt_mv_tpc=[]
    mv_withTopics=[]
    mv_annt_tpc=[]
    accu=[]
    rel_acc=[]
    '''
    (trueLabels,annt_tpc)=kappaInteragreemtWithTopics(annt_responses_local,nb_labels,tweets_topics_loc,groundTruth_temp_local)
    rel_acc.append(accuracyReliability(annt_topics_local,annt_tpc))
    accu.append(accuracy(trueLabels,groundTruth_temp_local,'Kappa-agreement with topics',annt_responses_local,tweets_topics_loc))
    print("annt_tpc kappa",annt_tpc)
    print("trueLabels kapppa")
    print(trueLabels)
    '''
    (mv_withTopics,mv_annt_tpc)=mvWithTopics(annt_responses_local,nb_topics,nb_labels,groundTruth_temp_local,tweets_topics_loc) 
    accu.append(accuracy(mv_withTopics,groundTruth_temp_local,'Majority Voting with Topics',annt_responses_local,tweets_topics_loc))
    rel_acc.append(accuracyReliability(annt_topics_local,mv_annt_tpc))
    print("mv with topics",mv_annt_tpc)
    print("mv_withTopics truelabe")
    print(mv_withTopics)
    '''
    (majority_voting,annt_mv_tpc)=majorityVoting(annt_responses_local,nb_labels,nb_topics,tweets_topics_loc)
    accu.append(accuracy(majority_voting,groundTruth_temp_local,'Majority Voting',annt_responses_local,tweets_topics_loc))
    rel_acc.append(accuracyReliability(annt_topics_local,annt_mv_tpc))
    
    print("majority voting",annt_mv_tpc)
    print("majority_voting")
    print(majority_voting)
    '''
    ##print(mv_annt_tpc)
    return (accu,rel_acc)

#===================Test===========================================================
def mainRunForIncremental(annt_responses_local,nb_labels,tweets_topics_local,groundTruth_temp_local,global_annt_tpc_Kappa,global_annt_tpc_MvTopics,global_annt_tpc_MV):
    nb_topics=len(tweets_topics_local[0])
    trueLabels=[]
    annt_tpc=[]
    majority_voting=[]
    annt_mv_tpc=[]
    mv_withTopics=[]
    mv_annt_tpc=[]
    accu=[]
    rel_acc=[]
    (trueLabels,annt_tpc)=kappaInteragreemtWithTopics(annt_responses_local,nb_labels,tweets_topics_local,groundTruth_temp_local)
    rel_acc.append(accumulateReliability(global_annt_tpc_Kappa,annt_tpc))
    accu.append(accuracyIncremental(trueLabels,groundTruth_temp_local,'Kappa-agreement with topics',annt_responses_local,tweets_topics_local))
    print(" incremental annt_tpc",annt_tpc)
    print("trueLabels kappa incrementa;",trueLabels)
    
    (mv_withTopics,mv_annt_tpc)=mvWithTopics(annt_responses_local,nb_topics,nb_labels,groundTruth_temp_local,tweets_topics_local) 
    accu.append(accuracyIncremental(mv_withTopics,groundTruth_temp_local,'Majority Voting with Topics',annt_responses_local,tweets_topics_local))
    rel_acc.append(accumulateReliability(global_annt_tpc_MvTopics,mv_annt_tpc))
    print("incremental mv with topics",mv_annt_tpc)
    print("mv_withTopics incremental truth",mv_withTopics)
    '''
    (majority_voting,annt_mv_tpc)=majorityVoting(annt_responses_local,nb_labels,nb_topics,tweets_topics_local)
    accu.append(accuracyIncremental(majority_voting,groundTruth_temp_local,'Majority Voting',annt_responses_local,tweets_topics_local))
    rel_acc.append(accumulateReliability(global_annt_tpc_MV,annt_mv_tpc))
    print("incremental majority voting",annt_mv_tpc)
    print("mv incremental ",majority_voting)
    '''   
    
    ##print(mv_annt_tpc)
    return (accu,rel_acc,global_annt_tpc_Kappa,global_annt_tpc_MvTopics,global_annt_tpc_MV)




#===========================================================================================
array_nb_topics=[15]
array_nb_annotators=[100]##[5,10,25,30]
array_maxFeatures=[2000]
array_midDf=[1]
array_maxiter=[700]
array_meanAccuarcy=[10]#,40]
array_SdAccuracy=[10]
array_meanLikelihood=[1]##,5,10,15]##,5,10,15]
array_nb_tweets=[1000]##[100,250,500,1000,5000]
nb_rounds=3
rel_annt_percent=0.4

step=100
result=[]
  
for nb_t in array_nb_topics:
 for mF in array_maxFeatures:
  for mD in array_midDf:
   for m in array_maxiter:
    for sdAcc in array_SdAccuracy:
     for meanAcc in array_meanAccuarcy:
      for nb_tweets in array_nb_tweets:
       for nb_a in array_nb_annotators:
        for meanLH in array_meanLikelihood:
     
         row_csv=[]
         row_csv.append(nb_tweets)
         row_csv.append(nb_t)
         row_csv.append(nb_a)
         row_csv.append("tweets.csv")
         row_csv.append(rel_annt_percent)
         
         ##dataset = pd.read_csv("finalizedfull.csv")
         dataset = pd.read_csv("Tweets.csv")
         data = dataset['text']
         ##data = dataset['tweet']
         vectorizer = TfidfVectorizer(max_features=mF, min_df=mD, stop_words='english')
         X = vectorizer.fit_transform(data)
         features1=vectorizer.get_feature_names()
         features=np.asarray(features1)
         nmf = NMF(n_components=nb_t, init='nndsvd', random_state=0, max_iter = m)
         nb_topics1=nb_t
         W=[]
         x=X.toarray()
         W1=[]
         W1 = nmf.fit_transform(X)
         H=nmf.components_
         topicsValue=np.full((mF,len(W1[0])),0.0)
         for i in range(0,mF):
          topicCounter=0
          theTopicValue=0
          theTopic=H[0][i]   
          for l in range(1,nb_topics1):
             if H[l][i]>theTopic:
                 theTopic=H[l][i]
                 theTopicValue=l
          topicsValue[i][theTopicValue]=theTopic
         oneTopicMatrix=np.full((len(W1),len(W1[0])),0.0)       
         for wordsCounter in range(0,mF):
            for tweetsCounter in range(0,nb_tweets):
                if x[tweetsCounter][wordsCounter]!=0.0:
                  for topicCounter in range(0,nb_t):
                   if topicsValue[wordsCounter][topicCounter]!=0.0:
                     oneTopicMatrix[tweetsCounter][topicCounter]=topicsValue[wordsCounter][topicCounter]
         '''             
         for n in range(0,10):
          for f in range(0,2000):
              if x[n][f]!=0:
                print(n,features[f]) 
                for l in range(0,nb_topics1):
                 if topicsValue[f][l]!=0.0:
                     print("topic",l)
         if topicCounter<3: 
               for i in range(0,nb_topics1):
                if H[i][j]!=0:
                   print(i,features[j])
          for i in range(0,25):
          print("topic",i)   
          for j in range(0,30):
              
              if topicsValue[j]==i:
                  print(features[j])
            
        
         for j in range (0,25):
             for i in range(0,100):
              if W1[i][j]!=0:
               print(data[i],j)
         '''     
         W=W1.copy()
         S=W.copy()
         ##ss=S[:nb_tweets]
         ss=oneTopicMatrix[:nb_tweets]
         tweets_topics=[]
         groundTruth_temp=[]
         documents = dataset[['airline_sentiment','text']]
         documents.replace({'neutral': 1, 'positive': 2, 'negative': 3}, inplace=True)
         groundTruth=documents['airline_sentiment']
         groundTruth=groundTruth.values
         twe_tpc=[]
         for i in range(0,len(ss)):
             v=0
             contain=False
             for j in range(0,len(ss[0])):
                 if ss[i,j]!=0.0:
                     contain=True
             if contain:
              v=v+1
              twe_tpc.append(ss[i])
              groundTruth_temp.append(groundTruth[i])
         print("v",v)
         
         tweets_topics=np.asarray(twe_tpc)
         tweets_topics=tweets_topics[range(0,nb_tweets),]
         groundTruth_temp=np.asarray(groundTruth_temp)
         groundTruth_temp=groundTruth_temp[range(0,nb_tweets),]
         nb_tweets=len(tweets_topics)
         print("tweets_topics",tweets_topics)
         nb_topics=len(tweets_topics[0])
         nb_annotators=nb_a
         nb_labels=3
         annt_responses=np.full((nb_a,nb_tweets),0)
         annt_topics=np.full((nb_a,nb_topics),1.0)
         trueLabels=[]
         topics=np.zeros(nb_topics)
         acc=np.zeros((1,3))
         acc=np.zeros((1,3))
         
         avg_accuracy_Kappa_interagreement=0.0
         avg_accuracy_MvWithTopics=0.0
         avg_accuracy_MV=0.0
         avg_rel_accuracy_Kappa_interagreement=0.0
         avg_rel_accuracy_MvWithTopics=0.0
         avg_rel_accuracy_MV=0.0
         
         avg_accuracy_Kappa_interagreement_NoRanking=0.0
         avg_accuracy_MvWithTopics_NoRanking=0.0
         avg_accuracy_MV_NoRanking=0.0
         avg_rel_accuracy_Kappa_interagreement_NoRanking=0.0
         avg_rel_accuracy_MvWithTopics_NoRanking=0.0
         avg_rel_accuracy_MV_NoRanking=0.0
         
         
         avg_accuracy_Kappa_interagreement_incremental=0.0
         avg_accuracy_MvWithTopics_incremental=0.0
         avg_accuracy_MV_incremental=0.0
         avg_rel_accuracy_Kappa_interagreement_incremental=0.0
         avg_rel_accuracy_MvWithTopics_incremental=0.0
         avg_rel_accuracy_MV_incremental=0.0
         
         avg_annotated_tweets=0
         avg_annotated_tweets_NoRanking=0
         avg_annotated_tweets_incremental=0
         
         for i in range (0,nb_rounds): 
             annotated_tweets=0
             annotated_tweets_NoRanking=0
             annotated_tweets_incremental=0
             
             accuracy_Kappa_interagreement=0.0
             accuracy_MvWithTopics=0.0
             accuracy_MV=0.0
             rel_accuracy_Kappa_interagreement=0.0
             rel_accuracy_MvWithTopics=0.0
             rel_accuracy_MV=0.0
             
             accuracy_Kappa_interagreement_NoRanking=0.0
             accuracy_MvWithTopics_NoRanking=0.0
             accuracy_MV_NoRanking=0.0
             rel_accuracy_Kappa_interagreement_NoRanking=0.0
             rel_accuracy_MvWithTopics_NoRanking=0.0
             rel_accuracy_MV_NoRanking=0.0
             annt_topics=[]
             annt_responses=[]
             annt_topicsFloat=[]
             
             (annt_topics,annt_topicsFloat,annt_responses)=simulateLabelsTopicsDependent(groundTruth_temp,tweets_topics,topics,nb_a,rel_annt_percent,sdAcc,meanLH,nb_labels)
             print ('annt_responses')
             print(annt_responses)
             print("groundTruth_temp")
             print(groundTruth_temp)
             print("annt_topics")
             print(annt_topics)
             

##======================================Main scenario====================================
             mv=[]
             acc=[]
             error_rel=[]
             annt_res_ordered=[]
             annt_res_ordered_tran=[]
             groundTruth_order=[]
             groundTruth_order_tran=[]
             mv_nb=np.zeros(nb_tweets)
             (mv,tp)=majorityVoting(annt_responses,nb_labels,nb_topics,tweets_topics)
             res_ord_list=[]
             tweets_topics_ord_list=[]
             tweets_topics_ordered=np.zeros(tweets_topics.shape)
             tweets_topics_order_tran=np.zeros(tweets_topics.shape)
             annt_res_ordered=np.zeros(annt_responses.shape)
             annt_res_ordered_tran=np.zeros(annt_responses.shape)
             label_ord_list=[]
             groundTruth_order=np.zeros((len(groundTruth_temp),1))
             groundTruth_order_tran=np.zeros((len(groundTruth_temp),1))
             for i in range(0,len(annt_responses[0])):
                 for j in range (0,len(annt_responses)):
                    if annt_responses[j][i]==mv[i] and mv[i]!=0:
                        mv_nb[i]=mv_nb[i]+1
             import math
             maximum=math.floor(mv_nb.max())
             ##print('mv_nb',mv_nb)
             for i in range(0,int(maximum)):
                 for j in range(0,len(mv_nb)):
                  if mv_nb[j]==(maximum-i) and mv_nb[j]!=0:
                      res_ord_list.append(annt_responses[:,j])
                      label_ord_list.append(groundTruth_temp[j])
                      tweets_topics_ord_list.append(tweets_topics[j,:])
                      ##print(j)
             annt_res_ordered=np.asarray(res_ord_list)
             annt_res_ordered_tran=annt_res_ordered.transpose()
             groundTruth_order=np.asarray(label_ord_list)
             groundTruth_order_tran= groundTruth_order.transpose()
             tweets_topics_ordered=np.asarray(tweets_topics_ord_list)
             
             print("annt_res_ordered")
             print(annt_res_ordered_tran)
             print("groundTruth_order_tran")
             print(groundTruth_order_tran)
             print("tweets_topics_ordered")
             print(tweets_topics_ordered)
             if annt_res_ordered_tran.size!=0:
               annotated_tweets=len(annt_res_ordered_tran[0]) 
               avg_annotated_tweets=annotated_tweets+avg_annotated_tweets
               (acc,error_rel)=mainRunWithSparsity(annt_res_ordered_tran,annt_topics,annt_topicsFloat,nb_labels,tweets_topics_ordered,groundTruth_order_tran)
               acc=np.asarray(acc)
               print(acc.shape)
               error_rel=np.asarray(error_rel)
               accuracy_Kappa_interagreement=accuracy_Kappa_interagreement+(acc[0])
               accuracy_MvWithTopics=accuracy_MvWithTopics+(acc[1])
               accuracy_MV=accuracy_MV+(acc[2])
               rel_accuracy_Kappa_interagreement=rel_accuracy_Kappa_interagreement+(error_rel[0])
               rel_accuracy_MvWithTopics=rel_accuracy_MvWithTopics+(error_rel[1])
               rel_accuracy_MV=rel_accuracy_MV+(error_rel[2])
               print(rel_accuracy_Kappa_interagreement,rel_accuracy_MvWithTopics,rel_accuracy_MV) 
               avg_accuracy_Kappa_interagreement=accuracy_Kappa_interagreement+avg_accuracy_Kappa_interagreement
               avg_accuracy_MvWithTopics=accuracy_MvWithTopics+avg_accuracy_MvWithTopics
               avg_accuracy_MV=accuracy_MV+avg_accuracy_MV
               avg_rel_accuracy_Kappa_interagreement=rel_accuracy_Kappa_interagreement+avg_rel_accuracy_Kappa_interagreement
               avg_rel_accuracy_MvWithTopics=rel_accuracy_MvWithTopics+avg_rel_accuracy_MvWithTopics
               avg_rel_accuracy_MV=rel_accuracy_MV+avg_rel_accuracy_MV
               print(avg_rel_accuracy_Kappa_interagreement,avg_rel_accuracy_MvWithTopics,avg_rel_accuracy_MV)
             
         
##=====================================End main scenario=====================================             


##=================================== without Ranking ===================================
             mv=[]
             acc_NoRanking=[]
             err_rel_NoRanking=[]
             
             annt_res_ordered=[]
             annt_res_ordered_tran=[]
             
             groundTruth_order=[]
             groundTruth_order_tran=[]
             
             tweets_topics_ord_list=[]
             tweets_topics_ordered=np.zeros(tweets_topics.shape)
             tweets_topics_order_tran=np.zeros(tweets_topics.shape)
             
             mv_nb=np.zeros(nb_tweets)
             (mv,tp)=majorityVoting(annt_responses,nb_labels,nb_topics,tweets_topics)
             res_ord_list=[]
             annt_res_ordered=np.zeros(annt_responses.shape)
             label_ord_list=[]
             groundTruth_order=np.zeros((len(groundTruth_temp),1))
             for i in range(0,len(annt_responses[0])):
                    if mv[i]!=0:
                        annt_res_ordered_tran.append(annt_responses[:,i])
                        groundTruth_order_tran.append(groundTruth_temp[i])
                        tweets_topics_ord_list.append(tweets_topics[i,:])
                        
             annt_res_ordered_tran=np.asarray(annt_res_ordered_tran) 
             annt_res_ordered_tran=annt_res_ordered_tran.transpose()
             groundTruth_order_tran=np.asarray(groundTruth_order_tran)
             groundTruth_order_tran=groundTruth_order_tran.transpose()
             annotated_tweets_NoRanking=len(annt_res_ordered_tran[0])
             tweets_topics_ordered=np.asarray(tweets_topics_ord_list)
             avg_annotated_tweets_NoRanking=avg_annotated_tweets_NoRanking+annotated_tweets_NoRanking
             (acc_NoRanking,err_rel_NoRanking)=mainRunWithSparsityNoranking(annt_res_ordered_tran,annt_topics,annt_topicsFloat,nb_labels,tweets_topics_ordered,groundTruth_order_tran)
             acc_NoRanking=np.asarray(acc_NoRanking)
             err_rel_NoRanking=np.asarray(err_rel_NoRanking)
             
             ##accuracy_Kappa_interagreement_NoRanking=accuracy_Kappa_interagreement_NoRanking+(acc_NoRanking[0])
             accuracy_MvWithTopics_NoRanking=accuracy_MvWithTopics_NoRanking+(acc_NoRanking[0])
             ##accuracy_MV_NoRanking=accuracy_MV_NoRanking+(acc_NoRanking[2])
             ##rel_accuracy_Kappa_interagreement_NoRanking=rel_accuracy_Kappa_interagreement_NoRanking+err_rel_NoRanking[0]
             rel_accuracy_MvWithTopics_NoRanking=rel_accuracy_MvWithTopics_NoRanking+err_rel_NoRanking[0]
             ##rel_accuracy_MV_NoRanking=rel_accuracy_MV_NoRanking+err_rel_NoRanking[2]
         
             ##avg_accuracy_Kappa_interagreement_NoRanking=accuracy_Kappa_interagreement_NoRanking+avg_accuracy_Kappa_interagreement_NoRanking
             avg_accuracy_MvWithTopics_NoRanking=accuracy_MvWithTopics_NoRanking+avg_accuracy_MvWithTopics_NoRanking
             ##avg_accuracy_MV_NoRanking=accuracy_MV_NoRanking+avg_accuracy_MV_NoRanking
             ##avg_rel_accuracy_Kappa_interagreement_NoRanking=rel_accuracy_Kappa_interagreement_NoRanking+avg_rel_accuracy_Kappa_interagreement_NoRanking
             avg_rel_accuracy_MvWithTopics_NoRanking=rel_accuracy_MvWithTopics_NoRanking+avg_rel_accuracy_MvWithTopics_NoRanking
             ##avg_rel_accuracy_MV_NoRanking=rel_accuracy_MV_NoRanking+avg_rel_accuracy_MV_NoRanking
             print(rel_accuracy_MV_NoRanking)        
             print(avg_rel_accuracy_MV_NoRanking) 
               
               
               
##=================================== End without Ranking==============================
             
             accuracy_Kappa_interagreement_incremental=0.0
             accuracy_MvWithTopics_incremental=0.0
             accuracy_MV_incremental=0.0
             rel_accuracy_Kappa_interagreement_incremental=0.0
             rel_accuracy_MvWithTopics_incremental=0.0
             rel_accuracy_MV_incremental=0.0
             sum_annotated_tweets_incremental=0.0
             err_rel_incremental=[]
             acc_incremental=[]
             batch=0
             nb_batches=0
             global_annt_topics_mv=np.full((len(annt_responses),len(tweets_topics[0])),1.0)
             global_annt_topics_mvTopic=np.full((len(annt_responses),len(tweets_topics[0])),1.0)
             global_annt_topics_kappa=np.full((len(annt_responses),len(tweets_topics[0])),1.0)
             while (batch<nb_tweets):
               
               tweets_topics1=tweets_topics[range(batch,batch+step),]
               groundTruth_temp1=groundTruth_temp[range(batch,batch+step),]
               annt_responses1=annt_responses[:,batch:batch+step]
               tweets_topics_ord_list=[]
               tweets_topics_ordered=np.zeros(tweets_topics1.shape)
               tweets_topics_order_tran=np.zeros(tweets_topics1.shape)  
               
               mv_nb=np.zeros(nb_tweets)
               print("incremental annt_responses1")
               print(annt_responses1)
               print("groundTruth_temp1")
               print(groundTruth_temp1)
               (mv,tp)=majorityVoting(annt_responses1,nb_labels,nb_topics,tweets_topics1)
               res_ord_list=[]
               annt_res_ordered=np.zeros(annt_responses1.shape)
               annt_res_ordered_tran=np.zeros(annt_responses1.shape)
               label_ord_list=[]
               groundTruth_order=np.zeros((len(groundTruth_temp1),1))
               groundTruth_order_tran=np.zeros((len(groundTruth_temp1),1))
               for i in range(0,len(annt_responses1[0])):
                  for j in range (0,len(annt_responses1)):
                        if annt_responses1[j][i]==mv[i] and mv[i]!=0:
                            mv_nb[i]=mv_nb[i]+1
               import math
               maximum=math.floor(mv_nb.max())
                 ##print('mv_nb',mv_nb)
               for i in range(0,int(maximum)):
                     for j in range(0,len(mv_nb)):
                      if mv_nb[j]==(maximum-i) and mv_nb[j]!=0:
                          res_ord_list.append(annt_responses1[:,j])
                          label_ord_list.append(groundTruth_temp1[j])
                          tweets_topics_ord_list.append(tweets_topics1[j,:])
                          ##print(j)
               annt_res_ordered=np.asarray(res_ord_list)
               annt_res_ordered_tran=annt_res_ordered.transpose()
               groundTruth_order=np.asarray(label_ord_list)
               groundTruth_order_tran= groundTruth_order.transpose()
               tweets_topics_ordered=np.asarray(tweets_topics_ord_list)
               print("tweets_topics_ordered")
               print(tweets_topics_ordered)
               if annt_res_ordered_tran.size!=0:
                   annotated_tweets_incremental=len(annt_res_ordered_tran[0])
                   ##print(annotated_tweets_incremental,"annotated_tweets_incremental")
                   sum_annotated_tweets_incremental=annotated_tweets_incremental+sum_annotated_tweets_incremental
                   ##print('here',sum_annotated_tweets_incremental,avg_annotated_tweets_incremental)
                   ##annt_topic=[]
                   ##annt_topicsFloat=[]
                   ##(annt_topics,annt_topicsFloat)=calculateReliability(groundTruth_order_tran,annt_res_ordered_tran,nb_topics,len(annt_res_ordered_tran[0]),tweets_topics_ordered)
                   (acc_incremental,err_rel_incremental,global_annt_topics_kappa,global_annt_topics_mvTopic,global_annt_topics_mv)=mainRunForIncremental(annt_res_ordered_tran,nb_labels,tweets_topics_ordered,groundTruth_order_tran,global_annt_topics_kappa,global_annt_topics_mvTopic,global_annt_topics_mv)
                   
                   acc_incremental=np.asarray(acc_incremental)
                   err_rel_incremental=np.asanyarray(err_rel_incremental)
                   accuracy_Kappa_interagreement_incremental=accuracy_Kappa_interagreement_incremental+(acc_incremental[0])
                   accuracy_MvWithTopics_incremental=accuracy_MvWithTopics_incremental+(acc_incremental[1])
                   ##accuracy_MV_incremental=accuracy_MV_incremental+(acc_incremental[2])
                   '''rel_accuracy_Kappa_interagreement_incremental=rel_accuracy_Kappa_interagreement_incremental+err_rel_incremental[0]
                   rel_accuracy_MvWithTopics_incremental=rel_accuracy_MvWithTopics_incremental+err_rel_incremental[1]
                   rel_accuracy_MV_incremental=rel_accuracy_MV_incremental+err_rel_incremental[2]
                   print("incremental",err_rel_incremental[2])'''
                   
                   nb_batches=nb_batches+1
               batch=batch+step    
             for row in range(0,len(annt_res_ordered_tran)):
                 for col in range(0,len(annt_topics[0])):
                     global_annt_topics_kappa[row,col]=global_annt_topics_kappa[row,col]-nb_batches
                     global_annt_topics_mvTopic[row,col]=global_annt_topics_mvTopic[row,col]-nb_batches
                     ##global_annt_topics_mv[row,col]=global_annt_topics_mv[row,col]-nb_batches
             print("incremetal kappa")
             print(global_annt_topics_kappa)
             avg_annotated_tweets_incremental=avg_annotated_tweets_incremental+sum_annotated_tweets_incremental
             accuracy_Kappa_interagreement_incremental=accuracy_Kappa_interagreement_incremental /sum_annotated_tweets_incremental   
             accuracy_MvWithTopics_incremental=accuracy_MvWithTopics_incremental/sum_annotated_tweets_incremental
             ##accuracy_MV_incremental=accuracy_MV_incremental/sum_annotated_tweets_incremental
             ##print(accuracy_Kappa_interagreement_incremental,accuracy_MvWithTopics_incremental,accuracy_MV_incremental)
             avg_accuracy_Kappa_interagreement_incremental=accuracy_Kappa_interagreement_incremental +avg_accuracy_Kappa_interagreement_incremental   
             avg_accuracy_MvWithTopics_incremental=accuracy_MvWithTopics_incremental+avg_accuracy_MvWithTopics_incremental
             ##avg_accuracy_MV_incremental=accuracy_MV_incremental+avg_accuracy_MV_incremental
             avg_rel_accuracy_Kappa_interagreement_incremental=avg_rel_accuracy_Kappa_interagreement_incremental+accuracyReliability(annt_topics,global_annt_topics_kappa)
             print("noor",avg_rel_accuracy_Kappa_interagreement_incremental)
             avg_rel_accuracy_MvWithTopics_incremental=avg_rel_accuracy_MvWithTopics_incremental+accuracyReliability(annt_topics,global_annt_topics_mvTopic)
             ##avg_rel_accuracy_MV_incremental=avg_rel_accuracy_MV_incremental+accuracyReliability(annt_topics,global_annt_topics_mv)
             
##====================================incremental train===============================================
                    
#=============End Incremental Train============================================================
         avg_accuracy_Kappa_interagreement=avg_accuracy_Kappa_interagreement/nb_rounds
         avg_accuracy_MvWithTopics=avg_accuracy_MvWithTopics/nb_rounds
         avg_accuracy_MV=avg_accuracy_MV/nb_rounds
         avg_rel_accuracy_Kappa_interagreement=avg_rel_accuracy_Kappa_interagreement/nb_rounds
         avg_rel_accuracy_MvWithTopics=avg_rel_accuracy_MvWithTopics/nb_rounds
         avg_rel_accuracy_MV=avg_rel_accuracy_MV/nb_rounds
         avg_annotated_tweets=avg_annotated_tweets/nb_rounds
         
         ##avg_accuracy_Kappa_interagreement_NoRanking=avg_accuracy_Kappa_interagreement_NoRanking/nb_rounds
         avg_accuracy_MvWithTopics_NoRanking=avg_accuracy_MvWithTopics_NoRanking/nb_rounds
         ##avg_accuracy_MV_NoRanking=avg_accuracy_MV_NoRanking/nb_rounds
         ##avg_rel_accuracy_Kappa_interagreement_NoRanking=avg_rel_accuracy_Kappa_interagreement_NoRanking/nb_rounds
         avg_rel_accuracy_MvWithTopics_NoRanking=avg_rel_accuracy_MvWithTopics_NoRanking/nb_rounds
         ##avg_rel_accuracy_MV_NoRanking=avg_rel_accuracy_MV_NoRanking/nb_rounds
         avg_annotated_tweets_NoRanking=avg_annotated_tweets_NoRanking/nb_rounds
         
         avg_accuracy_Kappa_interagreement_incremental=avg_accuracy_Kappa_interagreement_incremental/nb_rounds
         avg_accuracy_MvWithTopics_incremental=avg_accuracy_MvWithTopics_incremental/nb_rounds
         #avg_accuracy_MV_incremental=avg_accuracy_MV_incremental/nb_rounds
         avg_rel_accuracy_Kappa_interagreement_incremental=avg_rel_accuracy_Kappa_interagreement_incremental/nb_rounds
         avg_rel_accuracy_MvWithTopics_incremental=avg_rel_accuracy_MvWithTopics_incremental/nb_rounds
         #avg_rel_accuracy_MV_incremental=avg_rel_accuracy_MV_incremental/nb_rounds
         avg_annotated_tweets_incremental=avg_annotated_tweets_incremental/nb_rounds
         
         row_csv.append(avg_accuracy_Kappa_interagreement)
         row_csv.append(avg_rel_accuracy_Kappa_interagreement)
         row_csv.append(avg_accuracy_MvWithTopics)
         row_csv.append(avg_rel_accuracy_MvWithTopics)
         row_csv.append(avg_accuracy_MV)
         row_csv.append(avg_rel_accuracy_MV)
         row_csv.append(avg_annotated_tweets)
         row_csv.append("complete")
         my_file = Path('result1_file.csv')
         if my_file.exists():
             with open('result1_file.csv', 'a') as incsv:
              writer= csv.writer(incsv,lineterminator='\n')
              writer.writerow(row_csv)
              incsv.close()
         else:
             with open('result1_file.csv', 'w') as incsv:
              writer = csv.DictWriter(incsv, fieldnames = ["nb_tweets","nb_topics", "nb_annotators","Dataset","reliable annotators%","Kappa_agreement","Accuracy of Reliability","Majorty Voting topics","Accuracy of Reliability","MV","Accuracy of Reliability","Annotated tweets"])
              writer.writeheader()
              incsv.close() 
             with open('result1_file.csv', 'a') as incsv:
              writer= csv.writer(incsv,lineterminator='\n')
              writer.writerow(row_csv)
              incsv.close()
         
         row_csv=[]
         row_csv.append(nb_tweets)
         row_csv.append(nb_t)
         row_csv.append(nb_a)
         row_csv.append("tweets.csv")
         row_csv.append(rel_annt_percent)
         row_csv.append(avg_accuracy_Kappa_interagreement)
         row_csv.append(avg_rel_accuracy_Kappa_interagreement)
         row_csv.append(avg_accuracy_MvWithTopics_NoRanking)
         row_csv.append(avg_rel_accuracy_MvWithTopics_NoRanking)
         row_csv.append(avg_accuracy_MV)
         row_csv.append(avg_rel_accuracy_MV)
         row_csv.append(avg_annotated_tweets_NoRanking)
         row_csv.append("NoRanking")
         
         with open('result1_file.csv', 'a') as incsv:
              writer= csv.writer(incsv,lineterminator='\n')
              writer.writerow(row_csv)
              incsv.close()
              
         row_csv=[]
         row_csv.append(nb_tweets)
         row_csv.append(nb_t)
         row_csv.append(nb_a)
         row_csv.append("tweets.csv")
         row_csv.append(rel_annt_percent)     
         row_csv.append(avg_accuracy_Kappa_interagreement_incremental)
         row_csv.append(avg_rel_accuracy_Kappa_interagreement_incremental)
         row_csv.append(avg_accuracy_MvWithTopics_incremental)
         row_csv.append(avg_rel_accuracy_MvWithTopics_incremental)
         row_csv.append(avg_accuracy_MV)
         row_csv.append(avg_rel_accuracy_MV)
         row_csv.append(avg_annotated_tweets_incremental)
         row_csv.append("incremental")
         
         with open('result1_file.csv', 'a') as incsv:
              writer= csv.writer(incsv,lineterminator='\n')
              writer.writerow(row_csv)
              incsv.close()