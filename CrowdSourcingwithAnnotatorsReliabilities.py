# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 14:53:36 2018

@author: ASUS
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 14:26:57 2018

@author: Noor Jamaludeen
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from random import randrange
import csv
from pathlib import Path
import random

###======================================Topic dependent Simulator ==================================================
def simulateLabelsTopicsDependent(groundTruth,tweets_oneTopic,topics,numberAnnotators,reliablePercentage,SdAccuracy,meanLikelihood,nb_labels):
  likelihood=[]
  numberTopics=len(tweets_oneTopic[0])
  numberTweets=len(tweets_oneTopic)
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
       done=True
  counterreliable=0
  for m in range(0,numberAnnotators):
      counter=0
      is_reliable=np.random.binomial(1,reliablePercentage,1)
      if is_reliable:
       counterreliable=counterreliable+1   
       for i in range(0,numberTopics):
         topic_reliable=randrange(1,4,1)   
         if topic_reliable==1:       
             annt_oneTopic[m][i]=random.uniform(0.7,0.9)   
         else:
          if topic_reliable==2: 
             annt_oneTopic[m][i]=0.5   
          else:
             annt_oneTopic[m][i]=random.uniform(0.05,0.1) 
      else:
       for i in range(0,numberTopics):
         annt_oneTopic[m][i]=0.0  
      for tweetsCounter in range(0,numberTweets):
        correctProbability=0
        normalizeFactor=0
        annotate=np.random.binomial(1,likelihood[m],1)
        if (annotate[0]!=0.0):
         for topicCounter in range(0,nb_topics):
          if tweets_oneTopic[tweetsCounter][topicCounter]!=0:
           correctProbability=correctProbability+(tweets_oneTopic[tweetsCounter][topicCounter]*annt_oneTopic[m][topicCounter])
           normalizeFactor=normalizeFactor+tweets_oneTopic[tweetsCounter][topicCounter]
         if normalizeFactor!=0.0:   
          correctProbability=correctProbability/normalizeFactor
         else:
          correctProbability=0.0
         correct=np.random.binomial(1,correctProbability,1)
         counter=counter+1
         if correct[0]==1:   
          annt_responses[m,tweetsCounter]=groundTruth[tweetsCounter]
          for c in range(0,numberTopics):
                if tweets_oneTopic[tweetsCounter,c]!=0.0:
                    annt_topics[m,c]=annt_topics[m,c]+1
                    annt_topics_float[m,c]=round(annt_topics[m,c]+(tweets_oneTopic[tweetsCounter,c]),4)
         else:
           annt_responses[m,tweetsCounter]=randrange(1,nb_labels+1,1)  
           if annt_responses[m,tweetsCounter]==groundTruth[tweetsCounter]:
               for c in range(0,numberTopics):
                if tweets_oneTopic[tweetsCounter,c]!=0.0:
                    annt_topics[m,c]=annt_topics[m,c]+1
                    annt_topics_float[m,c]=round(annt_topics[m,c]+(tweets_oneTopic[tweetsCounter,c]),4)
           else:
               for c in range(0,numberTopics):
                if tweets_topics[tweetsCounter,c]!=0.0:
                    annt_topics[m,c]=annt_topics[m,c]-1
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
    kappa_agree_temp=np.zeros((len(annt_responses_local),len(annt_responses_local)))
    norms=np.full((len(annt_responses_local)),1.0)
    nb_annotators=len(annt_responses_local)
    nb_tweets=len(annt_responses_local[0])
    for i in range(0,nb_annotators):
     kappa=0.0
     for j in range(i+1,nb_annotators):
        common=False
        confusion=np.full((nb_labels,nb_labels),0)
        for z in range(0,nb_tweets):
          if annt_responses_local[i][z]!=0 and annt_responses_local[j][z]!=0:
                   common=True 
                   confusion[(annt_responses_local[i][z])-1][(annt_responses_local[j][z])-1]=confusion[(annt_responses_local[i][z])-1][(annt_responses_local[j][z])-1]+1
        if common==True:
            norms[i]=norms[i]+1
            norms[j]=norms[j]+1
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
        kappa_agree_temp[i,i]=1.0 
        if pre!=1: 
         kappa_agree_temp[i,j]=((pra-pre)/(1.0-pre))
         kappa_agree_temp[j,i]=((pra-pre)/(1.0-pre))
    kappa_agree_temp[nb_annotators-1,nb_annotators-1]=1.0
    for i1 in range(0,nb_annotators):
     kappa=0.0 
     for i2 in range(0,nb_annotators):
      kappa=kappa+kappa_agree_temp[i1,i2]
     kappa_agree.append(kappa/norms[i1]) 
    return (kappa_agree)
#===============End Kappa inter- agreement========================
#============Kappa without topics============================
def kappaInteragreemtWithTopics(annt_responses_local,nb_labels,tweets_topics_local,groundTruth_local):
    kappa_agree=kappa_aggreement(annt_responses_local,nb_labels)
    nb_annotators=len(annt_responses_local)
    nb_tweets=len(annt_responses_local[0])
    nb_topics=len(tweets_topics_local[0])
    annt_tpc_kappa=np.full((nb_annotators,nb_topics),1.0)
    Kappa_trueLabels=[]
    onlylabel=0.0
    for i in range(0,nb_tweets):
     highsim=-10000000.0
     truelabel=0
     for label in range(1,nb_labels+1):
         sim=0.0
         for j in range(0,nb_annotators):
             if annt_responses_local[j][i]==label:
                 onlylabel=label
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
              for c in range(0,nb_topics):
                  if tweets_topics_local[i,c]!=0.0:
                      annt_tpc_kappa[j,c]=annt_tpc_kappa[j,c]+1
      else:
             for c in range(0,nb_topics):
                  if tweets_topics_local[i,c]!=0.0 and annt_responses_local[j][i]!=0 :
                    annt_tpc_kappa[j,c]=annt_tpc_kappa[j,c]-1
    return(Kappa_trueLabels,annt_tpc_kappa)
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
                if (annt_responses_local[l][j]==majority) and (tweets_topics_local[j][c]!=0.0) :
                    annt_mv_tpc[l][c]=annt_mv_tpc[l][c]+1
                if (annt_responses_local[l][j]!=majority)and(annt_responses_local[l][j]!=0) and (tweets_topics_local[j][c]!=0.0) : 
                    annt_mv_tpc[l][c]=annt_mv_tpc[l][c]-1
    return (majority_voting,annt_mv_tpc)
##====================================== Our Brilliant Approach=======================================
def mvWithTopics(annt_responses_local,nb_topics,nb_labels,groundTruth_local,tweets_topics_local):
    nb_tweets=len(annt_responses_local[0])
    trueLabels=[]
    nb_annotators=len(annt_responses_local)
    annt_tpc=np.full((len(annt_responses_local),nb_topics),1.0)
    for i in range(0,nb_tweets):
     highsim=-10000000000.0
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
              for c in range(0,nb_topics):
                  if tweets_topics_local[i,c]!=0.0:
                   annt_tpc[k,c]=annt_tpc[k,c]+1
      else:
             for c in range(0,nb_topics):
                  if tweets_topics_local[i,c]!=0.0 and annt_responses_local[k][i]!=0:
                    annt_tpc[k,c]=annt_tpc[k,c]-1
    return (trueLabels,annt_tpc)
#============================ End Of Our Brillian Approach ======================================================== 
def accuracy(trueLabels_local,groundTruth_local,text,annt_responses,tweets_topics_local):
    hits=0
    nb_tweets=len(trueLabels_local)
    for i in range(0,nb_tweets):
     if groundTruth_local[i]==trueLabels_local[i]:
        hits=hits+1
    return (float(hits)/float(nb_tweets))

def accuracyIncremental(trueLabels_local,groundTruth_local,text,annt_responses,tweets_topics_local):
    hits=0
    nb_tweets=len(trueLabels_local)
    for i in range(0,nb_tweets):
     if groundTruth_local[i]==trueLabels_local[i]:
        hits=hits+1
    return (hits)


def accuracyReliability(realAnntTopics,estAnntTopics):
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
    (mv_withTopics,mv_annt_tpc)=mvWithTopics(annt_responses_local,nb_topics,nb_labels,groundTruth_temp_local,tweets_topics_loc) 
    accu.append(accuracy(mv_withTopics,groundTruth_temp_local,'Majority Voting with Topics',annt_responses_local,tweets_topics_loc))
    rel_acc.append(accuracyReliability(annt_topics_local,mv_annt_tpc))
    (majority_voting,annt_mv_tpc)=majorityVoting(annt_responses_local,nb_labels,nb_topics,tweets_topics_loc)
    accu.append(accuracy(majority_voting,groundTruth_temp_local,'Majority Voting',annt_responses_local,tweets_topics_loc))
    rel_acc.append(accuracyReliability(annt_topics_local,annt_mv_tpc))
    return (accu,rel_acc)
##==========================================================No Ranking======================================================
    
def mainRunWithSparsityNoranking(annt_responses_local,annt_topics_local,annt_topics_float,nb_labels,tweets_topics_loc,groundTruth_temp_local):
    nb_topics=len(annt_topics_local[0])
    mv_withTopics=[]
    mv_annt_tpc=[]
    accu=[]
    rel_acc=[]
    (mv_withTopics,mv_annt_tpc)=mvWithTopics(annt_responses_local,nb_topics,nb_labels,groundTruth_temp_local,tweets_topics_loc) 
    accu.append(accuracy(mv_withTopics,groundTruth_temp_local,'Majority Voting with Topics',annt_responses_local,tweets_topics_loc))
    rel_acc.append(accuracyReliability(annt_topics_local,mv_annt_tpc))
    return (accu,rel_acc)

#===================Test===========================================================
def mainRunForIncremental(annt_responses_local,nb_labels,tweets_topics_local,groundTruth_temp_local,global_annt_tpc_Kappa,global_annt_tpc_MvTopics,global_annt_tpc_MV):
    nb_topics=len(tweets_topics_local[0])
    trueLabels=[]
    annt_tpc=[]
    mv_withTopics=[]
    mv_annt_tpc=[]
    accu=[]
    rel_acc=[]
    (trueLabels,annt_tpc)=kappaInteragreemtWithTopics(annt_responses_local,nb_labels,tweets_topics_local,groundTruth_temp_local)
    rel_acc.append(accumulateReliability(global_annt_tpc_Kappa,annt_tpc))
    accu.append(accuracyIncremental(trueLabels,groundTruth_temp_local,'Kappa-agreement with topics',annt_responses_local,tweets_topics_local))
    (mv_withTopics,mv_annt_tpc)=mvWithTopics(annt_responses_local,nb_topics,nb_labels,groundTruth_temp_local,tweets_topics_local) 
    accu.append(accuracyIncremental(mv_withTopics,groundTruth_temp_local,'Majority Voting with Topics',annt_responses_local,tweets_topics_local))
    rel_acc.append(accumulateReliability(global_annt_tpc_MvTopics,mv_annt_tpc))
    return (accu,rel_acc,global_annt_tpc_Kappa,global_annt_tpc_MvTopics,global_annt_tpc_MV)

#===========================================================================================
array_nb_topics=[15]
array_nb_annotators=[500]
array_maxFeatures=[2000]
array_midDf=[1]
array_maxiter=[700]
array_meanAccuarcy=[10]
array_SdAccuracy=[10]
array_meanLikelihood=[1]
array_nb_tweets=[14640]
nb_rounds=3
rel_annt_percent=0.5
step=1464
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
         #row_csv.append("finalizedfull.csv")
         row_csv.append(rel_annt_percent)
         #dataset = pd.read_csv("finalizedfull.csv")
         dataset = pd.read_csv("Tweets.csv")
         data = dataset['text']
         #data = dataset['tweet']
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
         W=W1.copy()
         S=W.copy()
         ss=oneTopicMatrix[:nb_tweets]
         tweets_topics=[]
         tweets_topics1=[]
         groundTruth_temp=[]
         documents = dataset[['airline_sentiment','text']]
         documents.replace({'neutral': 1, 'positive': 2, 'negative': 3}, inplace=True)
         groundTruth=documents['airline_sentiment']
         '''
         documents = dataset[['senti','tweet']]
         documents.replace({0: 1, 4: 2, 2: 3}, inplace=True)
         groundTruth=documents['senti']
         '''
         groundTruth=groundTruth.values
         twe_tpc=[]
         v=0
         for i in range(0,len(ss)):
             contain=False
             for j in range(0,len(ss[0])):
                 if ss[i,j]!=0.0:
                     contain=True
             if contain:
              v=v+1
              twe_tpc.append(ss[i])
              groundTruth_temp.append(groundTruth[i])
         tweets_topics1=np.asarray(twe_tpc)
         nb_tweets=v
         tweets_topics=tweets_topics1[range(0,nb_tweets),]
         groundTruth_temp=np.asarray(groundTruth_temp)
         groundTruth_temp=groundTruth_temp[range(0,nb_tweets),]
         nb_tweets=len(tweets_topics)
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
         avg_annotated_tweets=0.0
         avg_annotated_tweets_NoRanking=0.0
         avg_annotated_tweets_incremental=0.0
         for i in range (0,nb_rounds): 
             annotated_tweets=0.0
             annotated_tweets_NoRanking=0.0
             annotated_tweets_incremental=0.0
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
             for i in range(0,int(maximum)):
                 for j in range(0,len(mv_nb)):
                  if mv_nb[j]==(maximum-i) and mv_nb[j]!=0:
                      res_ord_list.append(annt_responses[:,j])
                      label_ord_list.append(groundTruth_temp[j])
                      tweets_topics_ord_list.append(tweets_topics[j,:])
             annt_res_ordered=np.asarray(res_ord_list)
             annt_res_ordered_tran=annt_res_ordered.transpose()
             groundTruth_order=np.asarray(label_ord_list)
             groundTruth_order_tran= groundTruth_order.transpose()
             tweets_topics_ordered=np.asarray(tweets_topics_ord_list)
             if annt_res_ordered_tran.size!=0:
               annotated_tweets=len(annt_res_ordered_tran[0]) 
               avg_annotated_tweets=annotated_tweets+avg_annotated_tweets
               (acc,error_rel)=mainRunWithSparsity(annt_res_ordered_tran,annt_topics,annt_topicsFloat,nb_labels,tweets_topics_ordered,groundTruth_order_tran)
               acc=np.asarray(acc)
               error_rel=np.asarray(error_rel)
               accuracy_Kappa_interagreement=accuracy_Kappa_interagreement+(acc[0])
               accuracy_MvWithTopics=accuracy_MvWithTopics+(acc[1])
               accuracy_MV=accuracy_MV+(acc[2])
               rel_accuracy_Kappa_interagreement=rel_accuracy_Kappa_interagreement+(error_rel[0])
               rel_accuracy_MvWithTopics=rel_accuracy_MvWithTopics+(error_rel[1])
               rel_accuracy_MV=rel_accuracy_MV+(error_rel[2])
               avg_accuracy_Kappa_interagreement=accuracy_Kappa_interagreement+avg_accuracy_Kappa_interagreement
               avg_accuracy_MvWithTopics=accuracy_MvWithTopics+avg_accuracy_MvWithTopics
               avg_accuracy_MV=accuracy_MV+avg_accuracy_MV
               avg_rel_accuracy_Kappa_interagreement=rel_accuracy_Kappa_interagreement+avg_rel_accuracy_Kappa_interagreement
               avg_rel_accuracy_MvWithTopics=rel_accuracy_MvWithTopics+avg_rel_accuracy_MvWithTopics
               avg_rel_accuracy_MV=rel_accuracy_MV+avg_rel_accuracy_MV
         
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
             accuracy_MvWithTopics_NoRanking=accuracy_MvWithTopics_NoRanking+(acc_NoRanking[0])
             rel_accuracy_MvWithTopics_NoRanking=rel_accuracy_MvWithTopics_NoRanking+err_rel_NoRanking[0]
             avg_accuracy_MvWithTopics_NoRanking=accuracy_MvWithTopics_NoRanking+avg_accuracy_MvWithTopics_NoRanking
             avg_rel_accuracy_MvWithTopics_NoRanking=rel_accuracy_MvWithTopics_NoRanking+avg_rel_accuracy_MvWithTopics_NoRanking
               
##=================================== End without Ranking==============================
##====================================incremental train===============================================
             
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
             while (batch<v):    
               tweets_topics1=tweets_topics[range(batch,batch+step),]
               groundTruth_temp1=groundTruth_temp[range(batch,batch+step),]
               annt_responses1=annt_responses[:,batch:batch+step]
               tweets_topics_ord_list=[]
               tweets_topics_ordered=np.zeros(tweets_topics1.shape)
               tweets_topics_order_tran=np.zeros(tweets_topics1.shape)  
               mv_nb=np.zeros(nb_tweets)
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
               for i in range(0,int(maximum)):
                     for j in range(0,len(mv_nb)):
                      if mv_nb[j]==(maximum-i) and mv_nb[j]!=0:
                          res_ord_list.append(annt_responses1[:,j])
                          label_ord_list.append(groundTruth_temp1[j])
                          tweets_topics_ord_list.append(tweets_topics1[j,:])
               annt_res_ordered=np.asarray(res_ord_list)
               annt_res_ordered_tran=annt_res_ordered.transpose()
               groundTruth_order=np.asarray(label_ord_list)
               groundTruth_order_tran= groundTruth_order.transpose()
               tweets_topics_ordered=np.asarray(tweets_topics_ord_list)
               if annt_res_ordered_tran.size!=0:
                   annotated_tweets_incremental=len(annt_res_ordered_tran[0])
                   sum_annotated_tweets_incremental=annotated_tweets_incremental+sum_annotated_tweets_incremental
                   (acc_incremental,err_rel_incremental,global_annt_topics_kappa,global_annt_topics_mvTopic,global_annt_topics_mv)=mainRunForIncremental(annt_res_ordered_tran,nb_labels,tweets_topics_ordered,groundTruth_order_tran,global_annt_topics_kappa,global_annt_topics_mvTopic,global_annt_topics_mv)
                   acc_incremental=np.asarray(acc_incremental)
                   err_rel_incremental=np.asanyarray(err_rel_incremental)
                   accuracy_Kappa_interagreement_incremental=accuracy_Kappa_interagreement_incremental+(acc_incremental[0])
                   accuracy_MvWithTopics_incremental=accuracy_MvWithTopics_incremental+(acc_incremental[1])
                   nb_batches=nb_batches+1
               batch=batch+step    
             for row in range(0,len(annt_res_ordered_tran)):
                 for col in range(0,len(annt_topics[0])):
                     global_annt_topics_kappa[row,col]=global_annt_topics_kappa[row,col]-nb_batches
                     global_annt_topics_mvTopic[row,col]=global_annt_topics_mvTopic[row,col]-nb_batches
             avg_annotated_tweets_incremental=avg_annotated_tweets_incremental+sum_annotated_tweets_incremental
             accuracy_Kappa_interagreement_incremental=accuracy_Kappa_interagreement_incremental /sum_annotated_tweets_incremental   
             accuracy_MvWithTopics_incremental=accuracy_MvWithTopics_incremental/sum_annotated_tweets_incremental
             avg_accuracy_Kappa_interagreement_incremental=accuracy_Kappa_interagreement_incremental +avg_accuracy_Kappa_interagreement_incremental   
             avg_accuracy_MvWithTopics_incremental=accuracy_MvWithTopics_incremental+avg_accuracy_MvWithTopics_incremental
             avg_rel_accuracy_Kappa_interagreement_incremental=avg_rel_accuracy_Kappa_interagreement_incremental+accuracyReliability(annt_topics,global_annt_topics_kappa)
             avg_rel_accuracy_MvWithTopics_incremental=avg_rel_accuracy_MvWithTopics_incremental+accuracyReliability(annt_topics,global_annt_topics_mvTopic)
             
                    
#=============End Incremental Train============================================================
         avg_accuracy_Kappa_interagreement=avg_accuracy_Kappa_interagreement/nb_rounds
         avg_accuracy_MvWithTopics=avg_accuracy_MvWithTopics/nb_rounds
         avg_accuracy_MV=avg_accuracy_MV/nb_rounds
         avg_rel_accuracy_Kappa_interagreement=avg_rel_accuracy_Kappa_interagreement/nb_rounds
         avg_rel_accuracy_MvWithTopics=avg_rel_accuracy_MvWithTopics/nb_rounds
         avg_rel_accuracy_MV=avg_rel_accuracy_MV/nb_rounds
         avg_annotated_tweets=avg_annotated_tweets/nb_rounds
         avg_accuracy_MvWithTopics_NoRanking=avg_accuracy_MvWithTopics_NoRanking/nb_rounds
         avg_rel_accuracy_MvWithTopics_NoRanking=avg_rel_accuracy_MvWithTopics_NoRanking/nb_rounds
         avg_annotated_tweets_NoRanking=avg_annotated_tweets_NoRanking/nb_rounds
         avg_accuracy_Kappa_interagreement_incremental=avg_accuracy_Kappa_interagreement_incremental/nb_rounds
         avg_accuracy_MvWithTopics_incremental=avg_accuracy_MvWithTopics_incremental/nb_rounds
         avg_rel_accuracy_Kappa_interagreement_incremental=avg_rel_accuracy_Kappa_interagreement_incremental/nb_rounds
         avg_rel_accuracy_MvWithTopics_incremental=avg_rel_accuracy_MvWithTopics_incremental/nb_rounds
         avg_annotated_tweets_incremental=avg_annotated_tweets_incremental/nb_rounds
         row_csv.append(avg_accuracy_Kappa_interagreement)
         row_csv.append(avg_rel_accuracy_Kappa_interagreement)
         row_csv.append(avg_accuracy_MvWithTopics)
         row_csv.append(avg_rel_accuracy_MvWithTopics)
         row_csv.append(avg_accuracy_MV)
         row_csv.append(avg_rel_accuracy_MV)
         row_csv.append(avg_annotated_tweets)
         row_csv.append("complete")
         with open('result1_file_full_500_50.csv', 'w') as incsv:
              writer = csv.DictWriter(incsv, fieldnames = ["nb_tweets","nb_topics", "nb_annotators","Dataset","reliable annotators%","Kappa_agreement","Accuracy of Reliability","Majorty Voting topics","Accuracy of Reliability","MV","Accuracy of Reliability","Annotated tweets"])
              writer.writeheader()
              incsv.close() 
         with open('result1_file_full_500_50.csv', 'a') as incsv:
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
         with open('result1_file_full_500_50.csv', 'a') as incsv:
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
         with open('result1_file_full_500_50.csv', 'a') as incsv:
              writer= csv.writer(incsv,lineterminator='\n')
              writer.writerow(row_csv)
              incsv.close()