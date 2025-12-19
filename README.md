# Covid-19-Analysis
Repository that contains the analysis performed over a mexican database of people affected by covid-19. The analysis is focused on determine the risk factors linked to death related to covid. Tasks performed include data cleaning, normalization, standarization and codification, resampling, data visualization and model training (decease or survival classification). Model training was performed as final step to extract the importance of the features after classification was performed, as a explainability technique.

The notebook performs all the steps described in a sequential order, and there is no other requirement to run it or use it than previously install the imblearn, stats and squarify libraries. The only steps that are not sequential, are the model training, hyperparameter tuning, and k fold cross validations, as those cells executed several experiments or runs for each model type.

The project includes historical data from pandemic beginnings, from January to June 2020. The database used contains around 263000 cases in all Mexico. This database was collected with a centinel model. This model consisted in sampling the 10% of the patients that presented a viral-respiratory diagnosis for COVID-19. It comprehends the information reported by 475 hospitals, named USMER (Monitorization Units of Viral Respiratory Illnesses). The USMER were located all around the country, and comprehended all the health insurance or services given by the government (IMSS, ISSSTE, SEDENA, SEMAR). The database used is available in Kaggle [here](https://www.kaggle.com/code/marianarfranklin/mexico-positive-covid-19-cases-prediction). Data was accessed and stored in a csv file. It contained information on previous existing illnesses, age, sex, ambulatory patient, if the patient arrived UCI, and others. In total, in the database there exists 41 attributes.

The centinel model used for integrating the data was a process performed thanks to 'Secretaría de Salud'. The selected hospitals were because it was determined they are representative of all mexican population. The distribution of those units can be appreciated in the next image. For more details, refer to the [oficial page](https://www.gob.mx/salud/documentos/informacion-internacional-y-nacional-sobre-nuevo-coronavirus-2019-ncov). In the data, the attribute 'RESULTADO' with value 1, means a positive PCR and 2 a negative one. The data do not contain sensitive information, and it is free to the public to use.

<p align="center"><img width="500" height="400" alt="image" src="https://github.com/user-attachments/assets/c22c956f-5798-4f8a-b552-6a656fcdd983" /></p>

## Database Preprocessing
We performed a recodification to the flags defined in the original database. We detected null or inconsistent data, and eliminated them, achieving total cases of 260372. We also coded a variable for the death people, from the death date given, that was the only reference to someone's death. Age groups were also defined to mantain some groups of interest. The ranges for the groups were 0-15 (children), 15-25 (youths), 25-50 (adults), middle-aged (50-65) and elder (65+). New features were also added (quantity of previous illness and quantity of risk factors) during the analysis. Risk factors comprehended pregnancy, immunosuppressed and smoker.

There existed a huge proportion in the original database between the death and survived ones. The survivors were around the 98% of the total database. As we also intended to develop a classification model (this would be understood later), we explored both subsampling (Near Miss technique) and oversampling (SMOTE technique). The following image shows the distributions of the age feature with Near Miss and SMOTE applied. With near miss, we subsampled the survivors class from 255,962 to 22050 samples (approx, 8.6%), and with SMOTE we performed oversampling of the minority class from 4410 to achieve 17640 samples (approx. the 400%). 

<p align="center"><img width="450" height="600" alt="image" src="https://github.com/user-attachments/assets/7f3150ed-204c-4bc3-b11d-0403bcc65db7" /></p>

To determine if the sampling techniques were able to retain unaltered the data distribution, we analyzed the age distributions in both methods and compared statistical values and measurements. The next table illustrates those results, showing that the distributions remain unchanged. We also performed statistical proofs (Kolmogorov Smirnov, K-S) for a significance level of 5% to ensure to not alterate the original data distribution. Wassterstein distance was also measured for this comparison purposes and in another table the information of those results is provided.

|        | **Original data (survivors)** | **Resampling technique (survivors)**       | **Original data (deceased)** | **Subsampled class (deceased)**       |
|:--------:|:------------------------------:|:--------:|:------------------------------:|:--------:|
|   | **Original** | **Near Miss** | **Original** | **SMOTE** |             
| **Mean**      | 42.255 | 40.402  | 59.502  | 59.609  |
| **Median**    | 41     |39       |60       | 60      |
| **Mode**      | 30     |29       |52       |57       |
| **Bias**      | 0.267  | 0.3016  |-0.6693  |-0.6518  |
| **Kurtosis**  | 0.2146 |0.1803   |1.0869   | 1.045   |
| **Variance**  | 279.969| 296.63  |287.863  |253.93   |

| **Measure** | **Survivals vs survivals subsampled**       | **Deceased vs deceased oversampled** | **Survival vs deceased resampled**       |
|:--------:|:--------:|:--------:|:--------:|
| **K-S Statistic**             |0.0593  |0.02176  |0.4958 |
| **p-value**                   |1.66e-62| 0.0695  |0.0    |
| **Critical Value (D)**        |0.02289 | 0.02289 |0.02289|
| **Wassterstein Distance(W)**  |22.243  |13.714   |14.338 |
| **Divergence in distribution**| Samples could be from different samples but for practical effects it can be taken as from the same distribution (shift in distributions is small) | Samples from distributions come from the same distribution | Samples come from different distributions|

After the transformations to overcome class imbalance, we performed exploratory data analysis. All the variables except age were categorical, in particular, binary. A huge and continuous increment during march-april 2020 (when covid became pandemic and started to spread across the world) was noticed. There was also observed the states with more mortality, and we showed a graph containing the states that surpassed the national average of deaths in general. This information is also shown in the notebook. During the exploratory analysis, smoking was the risk factor more present in the death ones, and neumonia, diabetes, high blood pressure and obesity were the primary conditions that were more common in dead patients. In general, most patients were ambulatory in the hospitals, even with the oversampling of the minority class, and a small fraction of both survivals and dead ones were intubated.

## Model training and results

We then performed a classification in four classes: survivor with confirmed covid, survivor suspicious of covid, deceased with confirmed covid and deceased with suspicious covid. The confirmation was obtained with the PCR positive or negative test. Prior to model training, we performed scaling of the features (Min Max Scaler), and the presence or absence or certain risk factor or illness was treated as binary variable. We selected two popular models: decision tree and random forest classifiers. This was done because the feature importance can be returned once these models were trained, giving us an overview on how much affected the presence or absence of certain factors to survive or die.

Both models were trained with an 80-20 data partition of the balanced database. We also performed hyperparameter selection. To do so, a random search was performed in a predefined range for both classifiers. Once decided the best hyperparameters, k-fold cross validation with k=5 was performed for both cases. The metrics measured were accuracy, precision, recall and f2 score. F2 score was chosen because it weighs more the recall (capture or classify the deceased class accurately). In this case it is more important to perform less errors in confusing the deceased with survivors. The result of model performance metrics with k-fold is shown in the next image.

<p align="center"><img width="656" height="374" alt="image" src="https://github.com/user-attachments/assets/eabb162f-6a01-4b01-9c29-cc8667317d2e" /></p>

After the selection of the best performing model (random forest, for the ability to select better trees) we performed a feature importance analysis. This was performed extracting directly from the tree-based trained models. Pneumonia, diabetes, number of previous existing illness, age and intubated patients were the most importants features that the model used to classify deceased and survivors. Finally we defined a KPI named PFNC, as the proportion of deceased (either by covid or suspicious ones) that presented pneumonia, divided by the total cases of pneumonia. This KPI was intended to be monitored and hence, determine if a new variant of the disease (or other respiratory system diseases) should be considered as important as covid-19. This KPI measurement was of 80% in the resampled database and 7.5% in the original one, suggesting that values around or above this one, are of concern.








