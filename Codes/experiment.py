#!/usr/bin/env python
# coding: utf-8

from pickle import TRUE
import matplotlib.pyplot as plt
from typing import List
from scipy import stats
import TDSM_SOG
import numpy as np
from sklearn.metrics import silhouette_score
import pandas as pd
import researchpy as rp
from sklearn.model_selection import KFold
import experiment_TDSM
from scipy.stats import norm
from datetime import datetime
class Experiment():

        def __init__(self):
         return

        def __defaults__(self):
         return
        # smooth the curve
        
        def topology_som(self, som_num):
            start = int(np.sqrt(som_num))
            factor = som_num / start
            while not self.is_integer(factor):
                start += 1
                factor = som_num / start
            return int(factor), start
        
        def is_integer(self,number):
            if int(number) == number:
                return True
            else:
                return False
            

        def InitializedExperimentDataList(self,initial_neuron_num,
                                        dataread,
                                        all_test_score_tdsmsog_purity,
                                        all_test_score_tdsmsog_ari,
                                        all_test_score_tdsmsog_nmi,
                                        dim_num,
                                        train_indexes,
                                        test_indexes,
                                        n_granules,
                                      ):

              #nfolder= 3    
              purity_score_tdsmsog =[]
              ari_score_tdsmsog =[]
              nmi_score_tdsmsog =[]  
              tdsm_som_times=[]        
             # kfold = KFold(nfolder)
              for i in range(len(train_indexes)):
             # KFold(n_splits=nfolder, random_state=None, shuffle=False)
              #for i, (train_index, test_index) in enumerate(kfold.split(dataread.all_data)):
               # print(f" {i}  folder tdsmsog")
             #
              #purity_score_tdsmsog =[]
              #ari_score_tdsmsog =[]
              #nmi_score_tdsmsog =[]
                         
                start_time = datetime.now()

             # k = 0
              #for train, test in kfold.split(dataread.all_data):
                  # print(f" dataread.all_data { dataread.all_data.shape}")
                  
                  

                Comparision = TDSM_SOG.TDSMSOG(initial_neuron_num,
                        dataread,
                        dataread.all_label,
                        train_indexes[i],
                        test_indexes[i])     

                Comparision.do_SOGVSTDSMSOG(n_granules[i])
              
               # print(f"train_indexes[0] {train_indexes[0]}")
              #obtain sog encoding
            #  Comparision.train_data_embedding_tdsmsog
            #  Comparision.test_new_embedding_sog
              
              
                experiment = experiment_TDSM.Experiment()
             # self.granule = []
             # self.granule = experiment.TDSM(dataread,initial_neuron_num, dim_num )
            
                #if (i==0):
                 # print(f"Comparision.dim_num, {dim_num}")
                if(i == 0):
                 #print(11111)
                 combine_weights_sog_tdsm = experiment.TDSM2(Comparision.train_data_embedding_sog, Comparision.test_new_embedding_sog, initial_neuron_num, dim_num*len(n_granules[i]), dataread.all_label[train_indexes[i][0]],dataread.all_label[test_indexes[i][0]],True)
              
                else:
                  combine_weights_sog_tdsm = experiment.TDSM2(Comparision.train_data_embedding_sog, Comparision.test_new_embedding_sog, initial_neuron_num, dim_num*len(n_granules[i]), dataread.all_label[train_indexes[i][0]],dataread.all_label[test_indexes[i][0]])
              #combine_weights_sog_tdsm = experiment.TDSM2(dataread,Comparision.train_data_embedding_tdsmsog, Comparision.test_new_embedding_tdsmsog, som.m*som.n, dim_num* len(granule))

             # test_score_baseline_purity.append(Comparision.test_score_W0_p)
             # test_score_baseline_ari.append(Comparision.test_score_W0_a)
              #test_score_baseline_nmi.append(Comparision.test_score_W0_n)

              #return combine_weights_sog_tdsm.test_score_W0_p, combine_weights_sog_tdsm.test_score_W0_a,combine_weights_sog_tdsm.test_score_W0_n

               # if (i==0):
               #   print(f"combine_weights_sog_tdsm.test_score_W0_p, {combine_weights_sog_tdsm.test_score_W0_p}")

                
                purity_score_tdsmsog.append(combine_weights_sog_tdsm.test_score_W0_p)
                ari_score_tdsmsog.append(combine_weights_sog_tdsm.test_score_W0_a)
                nmi_score_tdsmsog.append(combine_weights_sog_tdsm.test_score_W0_n)
                end_time = datetime.now()
                seconds_difference =  (end_time - start_time).total_seconds()
                tdsm_som_times.append(seconds_difference)
              
            #  print(f"purity_score_tdsmsog {purity_score_tdsmsog}")
              self.confidient_interval("TDSMSOM","Purity",purity_score_tdsmsog)
              self.confidient_interval("TDSMSOM","ARI",ari_score_tdsmsog)
              self.confidient_interval("TDSMSOM","NMI",nmi_score_tdsmsog)
              print(f" tdsm-som  seconds_difference {np.mean(tdsm_som_times)} ")          
              if np.mean(purity_score_tdsmsog) > self.test_score_W0_p_max:
                  self.test_score_W0_p_max= np.mean(purity_score_tdsmsog) 
              if np.mean(nmi_score_tdsmsog) > self.test_score_W0_n_max:
                  self.test_score_W0_n_max= np.mean(nmi_score_tdsmsog) 
              if np.mean(ari_score_tdsmsog) > self.test_score_W0_a_max:
                   self.test_score_W0_a_max= np.mean(ari_score_tdsmsog) 
              
              all_test_score_tdsmsog_purity.append(np.mean(purity_score_tdsmsog))
              all_test_score_tdsmsog_nmi.append( np.mean(nmi_score_tdsmsog))
              all_test_score_tdsmsog_ari.append(np.mean(ari_score_tdsmsog))

        def confidient_interval(self,modelName,matriceName, matrices):
            mean = np.mean(matrices)
            std_dev = np.std(matrices, ddof=1)
            confidence_level = 0.95
            z = norm.ppf(1 - (1 - confidence_level) / 2) 
              # Calculate confidence interval
            margin_of_error = z * (std_dev / np.sqrt(len(matrices)))
            confidence_interval = (mean - margin_of_error, mean + margin_of_error)
            print(f"{modelName} Mean: {mean:.4f}")
            print(f"{modelName}  95% {matriceName} Confidence Interval: {confidence_interval}")
            
            
      

        def Ttest( self, dataread, initial_neuron_num, dim_num,scope_num,nfolder,PltName):
            
           # class_num = 9
           # dim_num = 11

         
 
            all_test_score_baseline_purity =[]
            all_test_score_baseline_ari =[]
            all_test_score_baseline_nmi =[]
          
            
            all_test_score_tdsmsog_purity =[]
            all_test_score_tdsmsog_ari =[]
            all_test_score_tdsmsog_nmi =[]
            
            plot_unit = [1]

           
            y = initial_neuron_num
            while y <= scope_num:
                experiment = experiment_TDSM.Experiment()
                print("experiment number: {}".format(y))        
                self.test_score_WTDSM_p_max = 0
                self.test_score_WTDSM_n_max = 0
                self.test_score_WTDSM_a_max = 0
    
                self.test_score_W0_p_max = 0
                self.test_score_W0_n_max = 0
                self.test_score_W0_a_max = 0
                print("TDSM !!!!!!!!!!!!!!")

                baseline_test_score_W0_p_list =[]
                baseline_test_score_W0_n_list =[]
                baseline_test_score_W0_a_list =[]
            
                kfold = KFold(nfolder)
                self.kfolders_index_train =[]
                self.kfolders_index_test =[]
                self.n_granules =[]
                self.tdsmNeurons=[]  
                tdsm_times =[]
             # KFold(n_splits=nfolder, random_state=None, shuffle=False)
                for i, (train_index, test_index) in enumerate(kfold.split(dataread.all_data)):
                  #print(f"dataread.all_data {dataread.all_data}")
                  #print(f" {i}  folder tdsm  train_index {train_index}test_index  {test_index}")
                 # print(f" train_index {train_index} dataread.all_label[train_index] {dataread.all_label[train_index]}") 
                  
                  start_time = datetime.now()

                  combine_weights_tdsm = experiment.TDSM2(dataread.all_data[train_index],dataread.all_data[test_index],initial_neuron_num, dim_num, dataread.all_label[train_index],dataread.all_label[test_index])
                  baseline_test_score_W0_p_list.append( combine_weights_tdsm.test_score_W0_p)
                  baseline_test_score_W0_n_list.append( combine_weights_tdsm.test_score_W0_n)
                  baseline_test_score_W0_a_list.append( combine_weights_tdsm.test_score_W0_a)
                  self.kfolders_index_train.append([train_index])
                  self.kfolders_index_test.append([test_index])
                  self.n_granules.append(combine_weights_tdsm.n_granule)
                  self.tdsmNeurons.append(len(combine_weights_tdsm.n_granule))
                  end_time = datetime.now()
                  seconds_difference =  (end_time - start_time).total_seconds()
                  tdsm_times.append(seconds_difference)
                  #print(f"combine_weights_tdsm.test_score_W0_p {combine_weights_tdsm.test_score_W0_p}")
                # predicted_clusters_indexes_tdsms.append(combine_weights_tdsm.)
                  #print(f" {i}  folder tdsm  seconds_difference {seconds_difference} ")
                self.confidient_interval("TDSM","Purity",baseline_test_score_W0_p_list )
                self.confidient_interval("TDSM","ARI",baseline_test_score_W0_a_list )
                self.confidient_interval("TDSM","NMI",baseline_test_score_W0_n_list )
                print(f" tdsm  seconds_difference {np.mean(tdsm_times)} ")
                print(f"tdsm perfection neuron numbers {np.mean(self.tdsmNeurons)}")          
                if np.mean(baseline_test_score_W0_p_list) > self.test_score_WTDSM_p_max:
                        self.test_score_WTDSM_p_max= np.mean(baseline_test_score_W0_p_list)
                if np.mean(baseline_test_score_W0_n_list) > self.test_score_WTDSM_n_max:
                        self.test_score_WTDSM_n_max=np.mean(baseline_test_score_W0_n_list)
                if np.mean(baseline_test_score_W0_a_list) > self.test_score_WTDSM_a_max:
                        self.test_score_WTDSM_a_max=np.mean(baseline_test_score_W0_a_list)

                all_test_score_baseline_purity.append(np.mean(baseline_test_score_W0_p_list))
               # print(f"(baseline_test_score_W0_p_list {(baseline_test_score_W0_p_list)}")
                all_test_score_baseline_ari.append(np.mean(baseline_test_score_W0_a_list))
                all_test_score_baseline_nmi.append(np.mean(baseline_test_score_W0_n_list))

                print("SOG TDSM !!!!!!!!!!!!!!")
              #  m, n = self.topology_som(y)
              #  som = newSom.SOM(m= m, n= n, dim=dim_num) 
                     
                self.InitializedExperimentDataList(initial_neuron_num,
                                        dataread,
                                        all_test_score_tdsmsog_purity,
                                        all_test_score_tdsmsog_ari,
                                        all_test_score_tdsmsog_nmi,   
                                        dim_num,
                                        self.kfolders_index_train,
                                        self.kfolders_index_test,
                                        self.n_granules           
                                        )        
                y =y +1
                if(y<= scope_num):
                    plot_unit.append(y)

               
            figure, axis = plt.subplots(1, 3,figsize =(12, 5))
            axis[0].set_title("Purity Score")               
            axis[1].set_title("ARI Score")
            axis[2].set_title("MNI Score") 


            print(f"best purity baseline  {self.test_score_WTDSM_p_max}")
            print(f"best ari  baseline {self.test_score_WTDSM_a_max}")
            print(f"best nmi  baseline{self.test_score_WTDSM_n_max}")
            
            print(f"best purity proposed {self.test_score_W0_p_max}")
            print(f"best ari proposed  {self.test_score_W0_a_max}")
            print(f"best nmi proposed {self.test_score_W0_n_max}")
   
          

            print(f"all_purity_score_baseline mean {np.mean(all_test_score_baseline_purity)}")
            print(f"all_ari_score_ba seline mean {np.mean(all_test_score_baseline_ari)}")
            print(f"all_nmi_score_baseline mean {np.mean(all_test_score_baseline_nmi)}")


            print(f"all_purity_score_tdsmsog mean {np.mean(all_test_score_tdsmsog_purity)}")
            print(f"all_ari_score_tdsmsog mean {np.mean(all_test_score_tdsmsog_ari)}")
            print(f"all_nmi_score_tdsmsog mean {np.mean(all_test_score_tdsmsog_nmi)}")





            axis[0].set_xlabel('Neuron number')
            axis[1].set_xlabel('Neuron number')
            axis[2].set_xlabel('Neuron number')

           # axis[3].set_xlabel('Experiment number')
            #axis[4].set_xlabel('Neuron number')

           # print(f"len plot_unit {len(plot_unit)}  len (all_test_score_baseline_accuracy) {len(all_test_score_baseline_purity)}")
            axis[0].plot(plot_unit,all_test_score_baseline_purity,'r',label ='baseline')
            axis[0].plot(plot_unit,all_test_score_tdsmsog_purity,'b',label ='proposed method')
            axis[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1)
         
            axis[1].plot(plot_unit,all_test_score_baseline_ari,'r',label ='baseline')
            axis[1].plot(plot_unit,all_test_score_tdsmsog_ari,'b',label ='proposed method')
            #axis[1].legend(loc='best')


            axis[2].plot(plot_unit,all_test_score_baseline_nmi,'r',label ='baseline')
            axis[2].plot(plot_unit,all_test_score_tdsmsog_nmi,'b',label ='proposed method')
          #  axis[2].legend(loc='best')



            
     

            plt.savefig(PltName + 'reslut.svg', format='svg', bbox_inches='tight')
            plt.show()
           
           #print(f"Normal test + all_test_score_baseline_purity")
           #shapiro_test = stats.shapiro(all_test_score_baseline_purity)
           #print(shapiro_test.pvalue)
           #print(f"Normal test + all_test_score_tdsmsog_purity")
           #shapiro_test = stats.shapiro(all_test_score_tdsmsog_purity)
           #print(shapiro_test.pvalue)

                                                    
                      
            df1 = pd.DataFrame(all_test_score_baseline_purity, columns = ['all_purity_score_baseline'])
            df2 = pd.DataFrame(all_test_score_tdsmsog_purity, columns = ['all_purity_score_tdsmsog'])

               
            print("Accuracy Score T-Test")
            t_statistic, p_value = stats.ttest_ind(df1,  df2) 
            #summary, results = rp.ttest(group1= df1['all_purity_score_baseline'], group1_name= "all_purity_score_baseline",
                                          #  group2= df2['all_purity_score_tdsmsog'], group2_name= "all_purity_score_tdsmsog")
            print(t_statistic)
            print(p_value)


           #print(f"Normal test + all_test_score_baseline_recall")
           #shapiro_test = stats.shapiro(all_test_score_baseline_ari)
           #print(shapiro_test.pvalue)
           #print(f"Normal test + all_test_score_fcg_recall")
           #shapiro_test = stats.shapiro(all_test_score_tdsmsog_ari)
           #print(shapiro_test.pvalue)


            df1 = pd.DataFrame(all_test_score_baseline_ari, columns = ['all_ari_score_baseline'])
            df2 = pd.DataFrame(all_test_score_tdsmsog_ari, columns = ['all_ari_score_tdsmsog'])

               
            print("ARI Score T-Test")
            t_statistic, p_value = stats.ttest_ind(df1,  df2) 
          #  summary, results = rp.ttest(group1= df1['all_ari_score_baseline'], group1_name= "all_ari_score_baseline",
                                #            group2= df2['all_ari_score_tdsmsog'], group2_name= "all_ari_score_tdsmsog")
            print(t_statistic)
            print(p_value)

           # print(f"Normal test + all_test_score_baseline_precision")
           # shapiro_test = stats.shapiro(all_test_score_baseline_nmi)
           # print(shapiro_test.pvalue)
           # print(f"Normal test + all_test_score_fcg_precision")
           # shapiro_test = stats.shapiro(all_test_score_tdsmsog_nmi)
           # print(shapiro_test.pvalue)
            
            df1 = pd.DataFrame(all_test_score_baseline_nmi, columns = ['all_nmi_score_baseline'])
            df2 = pd.DataFrame(all_test_score_tdsmsog_nmi, columns = ['all_nmi_score_tdsmsog'])

               
            print("NMI Score T-Test")
            t_statistic, p_value =stats.ttest_ind(df1,  df2) 
            #summary, results = rp.ttest(group1= df1['all_nmi_score_baseline'], group1_name= "all_nmi_score_baseline",
           #                                 group2= df2['all_nmi_score_tdsmsog'], group2_name= "all_nmi_score_tdsmsog")
            print(t_statistic)
            print(p_value)


        