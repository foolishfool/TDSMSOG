#!/usr/bin/env python
# coding: utf-8

from pickle import TRUE
import matplotlib.pyplot as plt
from typing import List
from scipy import stats
import FCG
import TDSM_SOG
import newSom
import numpy as np
from sklearn.metrics import silhouette_score
import pandas as pd
import researchpy as rp
from sklearn.model_selection import KFold
import experiment_TDSM

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
            

        def InitializedExperimentDataList(self,som,
                                        dataread,
                                        test_score_tdsmsog_purity,
                                        test_score_tdsmsog_ari,
                                        test_score_tdsmsog_nmi,
                                        granule,
                                        dim_num
                                      ):

           # if nfolder !=0:              
             # kfold = KFold(nfolder)
            #  KFold(n_splits=nfolder, random_state=None, shuffle=False)
              #for i, (train_index, test_index) in enumerate(kfold.split(dataread)):
              #purity_score_baseline =[]
              #ari_score_baseline =[]
              #nmi_score_baseline =[]
             #
              #purity_score_tdsmsog =[]
              #ari_score_tdsmsog =[]
              #nmi_score_tdsmsog =[]

             # k = 0
              #for train, test in kfold.split(dataread.all_data):
                  # print(f" dataread.all_data { dataread.all_data.shape}")
                  
                  

              Comparision = TDSM_SOG.TDSMSOG(som,
                        dataread.data_train,
                        dataread.data_test,          
                        dataread.label_train,
                        dataread.label_test)     

              Comparision.do_SOGVSTDSMSOG(granule)
              
              #obtain sog encoding
            #  Comparision.train_data_embedding_tdsmsog
            #  Comparision.test_new_embedding_sog
              
              
              experiment = experiment_TDSM.Experiment()
             # self.granule = []
             # self.granule = experiment.TDSM(dataread,initial_neuron_num, dim_num )
             # print(f"Comparision.train_data_embedding_tdsmsog shape {Comparision.train_data_embedding_tdsmsog.shape}")
              combine_weights_sog_tdsm = experiment.TDSM2(dataread,Comparision.train_data_embedding_sog, Comparision.test_new_embedding_sog, som.m*som.n, dim_num*som.m*som.n)
              #combine_weights_sog_tdsm = experiment.TDSM2(dataread,Comparision.train_data_embedding_tdsmsog, Comparision.test_new_embedding_tdsmsog, som.m*som.n, dim_num* len(granule))

             # test_score_baseline_purity.append(Comparision.test_score_W0_p)
             # test_score_baseline_ari.append(Comparision.test_score_W0_a)
              #test_score_baseline_nmi.append(Comparision.test_score_W0_n)

              #return combine_weights_sog_tdsm.test_score_W0_p, combine_weights_sog_tdsm.test_score_W0_a,combine_weights_sog_tdsm.test_score_W0_n

        
              if combine_weights_sog_tdsm.test_score_W0_p > self.test_score_W0_p_max:
                    self.test_score_W0_p_max= combine_weights_sog_tdsm.test_score_W0_p
              if combine_weights_sog_tdsm.test_score_W0_n > self.test_score_W0_n_max:
                    self.test_score_W0_n_max= combine_weights_sog_tdsm.test_score_W0_n
              if combine_weights_sog_tdsm.test_score_W0_a > self.test_score_W0_a_max:
                    self.test_score_W0_a_max= combine_weights_sog_tdsm.test_score_W0_a
              
              test_score_tdsmsog_purity.append(combine_weights_sog_tdsm.test_score_W0_p)
              test_score_tdsmsog_ari.append(combine_weights_sog_tdsm.test_score_W0_a)
              test_score_tdsmsog_nmi.append(combine_weights_sog_tdsm.test_score_W0_n)

                



        def Ttest( self, dataread, initial_neuron_num, dim_num,scope_num):
            
           # class_num = 9
           # dim_num = 11

            experiment = experiment_TDSM.Experiment()
            #self.granule =[]
            #self.granule = experiment.TDSM(dataread,initial_neuron_num, dim_num ).g_granule
           
           # print(f" self.test_score_W0_n { self.test_score_W0_n}")
           # print(f" self.test_score_W0_a { self.test_score_W0_a}")
            
            self.test_score_WTDSM_p_max = 0
            self.test_score_WTDSM_n_max = 0
            self.test_score_WTDSM_a_max = 0
 
            self.test_score_W0_p_max = 0
            self.test_score_W0_n_max = 0
            self.test_score_W0_a_max = 0
            
            combine_weights_tdsm = experiment.TDSM(dataread,initial_neuron_num, dim_num )
            self.granule = combine_weights_tdsm.g_granule
            print("TDSM !!!!!!!!!!!!!!")
            

            
            self.test_score_W0_p = combine_weights_tdsm.test_score_W0_p
            self.test_score_W0_n = combine_weights_tdsm.test_score_W0_n
            self.test_score_W0_a = combine_weights_tdsm.test_score_W0_a
                
            if combine_weights_tdsm.test_score_W0_p > self.test_score_WTDSM_p_max:
                    self.test_score_WTDSM_p_max= combine_weights_tdsm.test_score_W0_p
            if combine_weights_tdsm.test_score_W0_n > self.test_score_WTDSM_n_max:
                    self.test_score_WTDSM_n_max= combine_weights_tdsm.test_score_W0_n
            if combine_weights_tdsm.test_score_W0_a > self.test_score_WTDSM_a_max:
                    self.test_score_WTDSM_a_max= combine_weights_tdsm.test_score_W0_a
                    
            
            
            all_test_score_baseline_purity =[]
            all_test_score_baseline_ari =[]
            all_test_score_baseline_nmi =[]
          

            
            all_test_score_tdsmsog_purity =[]
            all_test_score_tdsmsog_ari =[]
            all_test_score_tdsmsog_nmi =[]
            
            
      
         #   m, n = self.topology_som(initial_neuron_num)
         #   som = newSom.SOM(m= m, n= n, dim=dim_num) 
         #            
         #   test_score_W1_p, test_score_W1_a,test_score_W1_n = self.InitializedExperimentDataList(som,
         #                               dataread,
         #                               all_test_score_tdsmsog_purity,
         #                               all_test_score_tdsmsog_ari,
         #                               all_test_score_tdsmsog_nmi,   
         #                               self.granule,
         #                               dim_num             
         #                               )     
         #   
         #   
         #   print(f" self.test_score_W1_p { test_score_W1_p}")
         #   print(f" self.test_score_W1_n { test_score_W1_n}")
         #   print(f" self.test_score_W1_a { test_score_W1_a}")
         #   
         #   
         #   if test_score_W1_p < self.test_score_W0_p:
         #    print("Not good purity result for discrete features !!!!!")
         #   if test_score_W1_n < self.test_score_W0_n:
         #    print("Not good nmi result for discrete features !!!!!")
         #   if test_score_W1_a < self.test_score_W0_a:
         #       print("Not good ari result for discrete features  !!!!!")
         #   
         #   return

            plot_unit = [1]
            
        

            y = initial_neuron_num
            while y <= scope_num:
                print("neuron number: {}".format(y))   
                
           # print(f" self.test_score_W0_p { self.test_score_W0_p}")
                
                
                all_test_score_baseline_purity.append(self.test_score_W0_p)
                all_test_score_baseline_ari.append(self.test_score_W0_a)
                all_test_score_baseline_nmi.append(self.test_score_W0_n)
                
                print("SOG TDSM !!!!!!!!!!!!!!")
                m, n = self.topology_som(y)
                som = newSom.SOM(m= m, n= n, dim=dim_num) 
                     
                self.InitializedExperimentDataList(som,
                                        dataread,
                                        all_test_score_tdsmsog_purity,
                                        all_test_score_tdsmsog_ari,
                                        all_test_score_tdsmsog_nmi,   
                                        self.granule,
                                        dim_num             
                                        )        
                y =y +2
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
            axis[0].plot(plot_unit,all_test_score_baseline_purity,'r',label ='all_purity_score_baseline')
            axis[0].plot(plot_unit,all_test_score_tdsmsog_purity,'b',label ='all_purity_score_tdsmsog')
            axis[0].legend(loc='best')

            axis[1].plot(plot_unit,all_test_score_baseline_ari,'r',label ='all_ari_score_baseline')
            axis[1].plot(plot_unit,all_test_score_tdsmsog_ari,'b',label ='all_ari_score_tdsmsog')
            axis[1].legend(loc='best')


            axis[2].plot(plot_unit,all_test_score_baseline_nmi,'r',label ='all_nmi_score_baseline')
            axis[2].plot(plot_unit,all_test_score_tdsmsog_nmi,'b',label ='all_nmi_score_tdsmsog')
            axis[2].legend(loc='best')



            
     


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
                
            summary, results = rp.ttest(group1= df1['all_purity_score_baseline'], group1_name= "all_purity_score_baseline",
                                            group2= df2['all_purity_score_tdsmsog'], group2_name= "all_purity_score_tdsmsog")
            print(summary)
            print(results)


           #print(f"Normal test + all_test_score_baseline_recall")
           #shapiro_test = stats.shapiro(all_test_score_baseline_ari)
           #print(shapiro_test.pvalue)
           #print(f"Normal test + all_test_score_fcg_recall")
           #shapiro_test = stats.shapiro(all_test_score_tdsmsog_ari)
           #print(shapiro_test.pvalue)


            df1 = pd.DataFrame(all_test_score_baseline_ari, columns = ['all_ari_score_baseline'])
            df2 = pd.DataFrame(all_test_score_tdsmsog_ari, columns = ['all_ari_score_tdsmsog'])

               
            print("ARI Score T-Test")
                
            summary, results = rp.ttest(group1= df1['all_ari_score_baseline'], group1_name= "all_ari_score_baseline",
                                            group2= df2['all_ari_score_tdsmsog'], group2_name= "all_ari_score_tdsmsog")
            print(summary)
            print(results)

           # print(f"Normal test + all_test_score_baseline_precision")
           # shapiro_test = stats.shapiro(all_test_score_baseline_nmi)
           # print(shapiro_test.pvalue)
           # print(f"Normal test + all_test_score_fcg_precision")
           # shapiro_test = stats.shapiro(all_test_score_tdsmsog_nmi)
           # print(shapiro_test.pvalue)
            
            df1 = pd.DataFrame(all_test_score_baseline_nmi, columns = ['all_nmi_score_baseline'])
            df2 = pd.DataFrame(all_test_score_tdsmsog_nmi, columns = ['all_nmi_score_tdsmsog'])

               
            print("NMI Score T-Test")
                
            summary, results = rp.ttest(group1= df1['all_nmi_score_baseline'], group1_name= "all_nmi_score_baseline",
                                            group2= df2['all_nmi_score_tdsmsog'], group2_name= "all_nmi_score_tdsmsog")
            print(summary)
            print(results)


  

