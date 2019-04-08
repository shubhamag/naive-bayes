
# coding: utf-8

# In[13]:


import pandas


# In[14]:


data_path = '/home/shubham/Studies/10-701/hw2' + '/adult.data'


# In[15]:


col_names = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race',
            'sex','capital-gain','capital-loss','hours-per-week','native-country','income']
sep = ',\s'
usecols = ''


# In[16]:


df = pandas.read_table(data_path,sep = sep,names=col_names)
df = df[~(df == '?').any(axis=1)]


# In[17]:


rich_data = df[df['income']=='>50K']
poor_data = df[df['income']=='<=50K']


# In[18]:


rich_mean = dict(rich_data.mean(axis=0))
rich_std = dict(rich_data.std(axis=0))
poor_mean = dict(poor_data.mean(axis=0))
poor_std = dict(poor_data.std(axis=0))


# In[19]:


poor_education = dict(poor_data['education'].value_counts())
poor_country = dict(poor_data['native-country'].value_counts())
poor_occupation = dict(poor_data['occupation'].value_counts())
poor_relationship = dict(poor_data['relationship'].value_counts())
poor_marital = dict(poor_data['marital-status'].value_counts())
poor_race = dict(poor_data['race'].value_counts())
poor_sex = dict(poor_data['sex'].value_counts())
poor_workclass = dict(poor_data['workclass'].value_counts())
poor_dict = {'education':poor_education,'native-country':poor_country,'occupation':poor_occupation,'sex':poor_sex,
          'race':poor_race,'workclass':poor_workclass,'relationship':poor_relationship,'marital-status':poor_marital}

rich_education = dict(rich_data['education'].value_counts())
rich_country = dict(rich_data['native-country'].value_counts())
rich_occupation = dict(rich_data['occupation'].value_counts())
rich_relationship = dict(rich_data['relationship'].value_counts())
rich_marital = dict(rich_data['marital-status'].value_counts())
rich_race = dict(rich_data['race'].value_counts())
rich_sex = dict(rich_data['sex'].value_counts())
rich_workclass = dict(rich_data['workclass'].value_counts())
rich_dict = {'education':rich_education,'native-country':rich_country,'occupation':rich_occupation,'sex':rich_sex,
          'race':rich_race,'workclass':rich_workclass,'relationship':rich_relationship,'marital-status':rich_marital}


# In[25]:


rich_count = rich_data.shape[0]
poor_count = poor_data.shape[0]
total_count   = rich_count+poor_count
def print_alphas(d,den,keys):
    output = ""
    for key in keys:
        if key in d:
            output+=key + "="+ "%.4f"%(d[key]/den) +" , "
    return output


# In[26]:


print("Prior probability of y, P(y)\n")
print(" P(y = '>50K'):",rich_count/total_count)
print(" P(y = '<=50K'):",poor_count/total_count)


# In[22]:


print(rich_workclass)


# In[28]:


workclass_order = [ 'Private','Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 
                   'Never-worked']
education_order = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc',
                   '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool']
marital_order = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated',
                 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse' ]
occupation_order = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales',
                    'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct',
                    'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv',
                    'Protective-serv', 'Armed-Forces']
relationship_order = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']

race_order = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
sex_order = ['Female','Male']

country_order = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany',
                 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China',
                 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland',
                 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France',
                 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala',
                 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador',
                 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']


print("Class > 50K\n")
print("age: mean=",rich_mean['age']," , std-dev=" ,rich_std['age'])
print("workclass: ", print_alphas(rich_workclass,rich_count,workclass_order))
print("fnlgwt: mean=",rich_mean['fnlwgt']," , std-dev=" ,rich_std['fnlwgt'])
print("education: ", print_alphas(rich_education,rich_count,education_order))
print("education-num: mean=",rich_mean['education-num']," , std-dev=" ,rich_std['education-num'])
print("marital-status: ", print_alphas(rich_marital,rich_count,marital_order))
print("occupation: ", print_alphas(rich_occupation,rich_count,occupation_order))
print("relationship: ", print_alphas(rich_relationship,rich_count,relationship_order))
print("race: ", print_alphas(rich_race,rich_count,race_order))
print("sex: ", print_alphas(rich_sex,rich_count,sex_order))
print("capital-gain: mean=",rich_mean['capital-gain']," , std-dev=" ,rich_std['capital-gain'])
print("capital-loss: mean=",rich_mean['capital-loss']," , std-dev=" ,rich_std['capital-loss'])
print("hours-per-week: mean=",rich_mean['hours-per-week']," , std-dev=" ,rich_std['hours-per-week'])
print("country: ", print_alphas(rich_country,rich_count,country_order))

print("\n\n")
print("Class <= 50K\n")
print("age: mean=",poor_mean['age']," , std-dev=" ,poor_std['age'])
print("workclass: ", print_alphas(poor_workclass,poor_count,workclass_order))
print("fnlgwt: mean=",poor_mean['fnlwgt']," , std-dev=" ,poor_std['fnlwgt'])
print("education: ", print_alphas(poor_education,poor_count,education_order))
print("education-num: mean=",poor_mean['education-num']," , std-dev=" ,poor_std['education-num'])
print("marital-status: ", print_alphas(poor_marital,poor_count,marital_order))
print("occupation: ", print_alphas(poor_occupation,poor_count,occupation_order))
print("relationship: ", print_alphas(poor_relationship,poor_count,relationship_order))
print("race: ", print_alphas(poor_race,poor_count,race_order))
print("sex: ", print_alphas(poor_sex,poor_count,sex_order))
print("capital-gain: mean=",poor_mean['capital-gain']," , std-dev=" ,poor_std['capital-gain'])
print("capital-loss: mean=",poor_mean['capital-loss']," , std-dev=" ,poor_std['capital-loss'])
print("hours-per-week: mean=",poor_mean['hours-per-week']," , std-dev=" ,poor_std['hours-per-week'])
print("country: ", print_alphas(poor_country,poor_count,country_order))


# In[60]:


dict(rich_data.iloc[2])
import math
discrete = [k for k in rich_dict]

def norm_prob(x,mean,var):
    eps = 1e-9
    num = -((x - mean)**2) / (2*(var+eps))
    p = math.e**num
    p = p/ math.sqrt(2*math.pi*(var+eps))
    return p
    
def log_posteriors(d):
    log_p = [0,0]
    p =0
    
    prior = rich_count/total_count
    p +=math.log(prior)
    for key,value in d.items():
            if(key=='income'):
                continue
        
            if key in discrete:
                try:
                    alpha = rich_dict[key][value] / rich_count
                    p+=math.log(alpha)
                except:
                    p+= -100
            else:
                try:
                
                    p+=math.log(norm_prob(value,rich_mean[key],rich_std[key]**2))
                except:
                    print("error in rich",key)
    log_p[1] = p
    prior = poor_count/total_count
    p =math.log(prior)
    for key,value in d.items():
            if(key=='income'):
                continue
        
            if key in discrete:
                try:
                    alpha = poor_dict[key][value] / poor_count
                    p+=math.log(alpha)
                except:
                    p+=-100
            else:
                try:
                
                    p+=math.log(norm_prob(value,poor_mean[key],poor_std[key]**2))
                except:
                    print("error in poor",key)
    log_p[0] = p
                

    return log_p
    
    


# In[30]:


log_posteriors(dict(rich_data.iloc[2]))


# In[31]:


test_path = '/home/shubham/Studies/10-701/hw2' + '/adult.test'
tf = pandas.read_table(test_path,sep = sep,names=col_names,skiprows=[0])

tf = tf[~(tf == '?').any(axis=1)]
tf.iloc[0]
# log_posteriors(dict(tf.iloc[1]))
# log_posteriors(dict(tf.iloc[2]))
# log_posteriors(dict(tf.iloc[3]))


# In[32]:


print(log_posteriors(dict(tf.iloc[0])))
print(log_posteriors(dict(tf.iloc[1])))
print(log_posteriors(dict(tf.iloc[2])))
log_posteriors(dict(tf.iloc[3]))


# In[57]:


import re
def classify(tf):
    correct = 0
    count = tf.shape[0]
    for i in range(count):
        if(i%100 ==0):
            print(i,correct)
        d = dict(tf.iloc[i])
        l=[10,0]
#         try:
        l = log_posteriors(d)
#         except:
#             print("Error at index:",i)
            
        if(l[0] > l[1]):
            pred = '<=50K.'
        else:
            pred = '>50K.'
        if(re.match(d['income'],pred)):
            correct+=1
#             print(i,": correct")
        else:
            pass
#             print(d['income'], ' guessed: ',pred)
    return correct,correct/count
            
            


# In[41]:


d = dict(tf.iloc[3])
d['income']
classify(tf)


# In[43]:


classify(df)


# In[61]:


n = [2**i for i in range(5,14)]
train_accuracies= []
test_accuracies= []
for j in n:
    df = pandas.read_table(data_path,sep = sep,names=col_names,nrows = j)
    df = df[~(df == '?').any(axis=1)]
    
    #remove ?
    rich_data = df[df['income']=='>50K']
    poor_data = df[df['income']=='<=50K']

    rich_mean = dict(rich_data.mean(axis=0))
    rich_std = dict(rich_data.std(axis=0))
    poor_mean = dict(poor_data.mean(axis=0))
    poor_std = dict(poor_data.std(axis=0))

    poor_education = dict(poor_data['education'].value_counts())
    poor_country = dict(poor_data['native-country'].value_counts())
    poor_occupation = dict(poor_data['occupation'].value_counts())
    poor_relationship = dict(poor_data['relationship'].value_counts())
    poor_marital = dict(poor_data['marital-status'].value_counts())
    poor_race = dict(poor_data['race'].value_counts())
    poor_sex = dict(poor_data['sex'].value_counts())
    poor_workclass = dict(poor_data['workclass'].value_counts())
    poor_dict = {'education':poor_education,'native-country':poor_country,'occupation':poor_occupation,'sex':poor_sex,
          'race':poor_race,'workclass':poor_workclass,'relationship':poor_relationship,'marital-status':poor_marital}

    rich_education = dict(rich_data['education'].value_counts())
    rich_country = dict(rich_data['native-country'].value_counts())
    rich_occupation = dict(rich_data['occupation'].value_counts())
    rich_relationship = dict(rich_data['relationship'].value_counts())
    rich_marital = dict(rich_data['marital-status'].value_counts())
    rich_race = dict(rich_data['race'].value_counts())
    rich_sex = dict(rich_data['sex'].value_counts())
    rich_workclass = dict(rich_data['workclass'].value_counts())
    rich_dict = {'education':rich_education,'native-country':rich_country,'occupation':rich_occupation,'sex':rich_sex,
          'race':rich_race,'workclass':rich_workclass,'relationship':rich_relationship,'marital-status':rich_marital}
    #split rcih poor
    
    #copy rich poor dict defs
    correct,acc = classify(tf)
    test_accuracies.append(acc)
    correct,acc = classify(df)
    train_accuracies.append(acc)



# In[70]:


test_acc = [100*t for t in test_accuracies]
train_acc = [100*t for t in train_accuracies]
print(test_acc)
print(train_acc)
import matplotlib.pyplot as plt
plt.figure(1)
plt.subplot(211)
plt.plot(n, train_acc, linewidth=2.0)

plt.subplot(212)
plt.plot(n,test_acc,linewidth=2.0)
plt.show()


