# the thidf module for hddm
# pipreqs  --encoding=utf-8 --force ./

class HDDMs():
  """
    In order to process multiple models

    vars: _ms: save models
    func: loadm：could load models list, models dict or path

  """

  def __init__(self,*ms):
    self.name = "HDDM models"
    self._ms = {}
    self._ms = self.__check_ms(*ms)
    
  def __check_ms(self,*ms):
    if ms != ():
      try:
        if isinstance(ms[0],dict):
          print("ms is dict")
        elif isinstance(ms[0],list):
          msn = [j for j in range(len(ms[0]))]
          msv = ms[0]
      except:
        print('ms is wrong')
    else:
      if self._ms == {}:
        print("ms is not exist")
        print("attempt to load local model files")
        try:
          ms = self.loadm()  
        except :
          print('load failed')
      else:
        ms = self._ms
        
      msn = ms.keys()
      msv = ms.values()
      return msn,msv,ms

  def loadm(self,name="*"):
    """
    加载模型
    :param: name:str, 模型名称，可选参数，如果没有将加载目录下所有模型
    :return: dict, keys为模型名称，value为HDDM模型 
    """
    ms={}
    import glob
    import hddm
    if isinstance(name,list):
      files = [glob.glob(i+".hddm") for i in name]
    elif isinstance(name,str):
      files = glob.glob(name+".hddm")
    else:
      print("para name is in wrong type")
    
    for i in files:
      ms[i.split(".",1)[0]] = hddm.load(i)
    if self._ms == {}:
      print("load new model")
      self._ms = ms
    else:
      print('add model into ms')
      self._ms.update(ms)
    return ms

  def DIC_results(self,*ms): 
    """
    计算所有模型DIC, 方便进行模型比较

    :param ms:包含hddm的字典或者列表, 可选参数，默认将自动加载目录下所有模型并计算DIC
    :return: dataframe + hddms
    """
    import pandas as pd
    msn,msv = self.__check_ms(*ms)
    results = pd.DataFrame({"m_name":msn,"DIC":[i.dic for i in msv]})
    return results.sort_values("DIC",ascending=False, inplace=False),ms

  def params(self,*ms):
    """
    计算所有模型group层面的参数值，以便于之后做差异检验

    :param ms:hddm, 模型，可选参数，如果没有将自动加载目录下所有模型
    :return: dataframe, long-data format
    """
    import pandas as pd

    _,_,ms = self.__check_ms(*ms)
    # 返回dataframe list
    rs_p = []
    dbn_p = {}
    for i,j in ms.items():
      rs = j.get_group_nodes().copy()
      rs["name"] = i
      rs_p.append(rs)

      dbn_p[i] = j.nodes_db.node

    rs_ps = pd.concat(rs_p,join='inner',axis=0) # 将list 合成一个 dataframe
    
    self.pr = rs_ps # params resluts
    self.dbs = dbn_p # params db data

    return rs_ps,dbn_p

  def get_wdi(self,mname='m',save=False):
    """ Get wide data format for individual models parameters 
    """
    import pandas as pd
    temp = self._ms[mname].get_subj_nodes().loc[:,['mean','subj_idx','node']].reset_index(drop=True)
    temp1 = temp[~pd.isna(temp['mean'])].copy()
    temp1['node'] = [str(i).split(')')[0] for i in temp1['node'].values]
    temp2 = pd.pivot_table(temp1,index=['subj_idx'],columns=['node'],values=['mean'])
    temp3 = temp2.droplevel(0,axis=1).reset_index().copy()
    temp3 = temp3.convert_dtypes()

    if save:
      temp3.to_csv('mname'+'csv')

    return temp3


def contrast(params,name=0,condinum=2):
  """
  所有参数avtz，的差异性检验
  :param params:模型参数，来自于函数params
  :param name:模型名字, 必须指定需要比较的模型
  :param condinum:必须指明变量有多少水平
  :return:p，f值
  """
  import hddm
  import matplotlib as plt
  if isinstance(name,int):
    mname = params["modelname"][name]
    para = params["para_all"][name]
  else:
    mname = name
    para = params["para_all"][params["modelname"] == name]

  para = para[para.index.str.contains('.{6}\(')]
  paraname = ['a','v','t','z']
  para = {i:para[para.index.str.contains(i)] for i in paraname}

  results = {}
  for i,j in para.items():
    if len(j) == 0:
      continue
    a = [j[j.index.str.contains('\('+str(i+1)+'\)')]['mean'] for i in range(condinum)]
    from scipy.stats import f_oneway
    if condinum == 2:
      f,p = f_oneway(a[0],a[1])
    elif condinum == 3:
      f,p = f_oneway(a[0],a[1],a[2])
    elif condinum == 4:
      f,p = f_oneway(a[0],a[1],a[2],a[3])
    elif condinum == 5:
      f,p = f_oneway(a[0],a[1],a[2],a[3],a[4])
    elif condinum == 6:
      f,p = f_oneway(a[0],a[1],a[2],a[3],a[4],a[5])
    results[mname+str(i)] = (f,p)
  return results

def contrast_plot(m,name=0,paralist=['v','a','t','z'],condnum=2):
  """
  参数差异比较画图
  :param m:hddm模型
  :param name:模型名称，或者数字索引
  :param paralist:list,指定需要比较的参数
  :param condnum:int,条件数量
  :return:dataframe + hddm
  """
  import hddm 
  import matplotlib.pyplot as plt

  condition = ["("+str(x+1)+")" for x in range(condnum)]

  if isinstance(name,int):
    mm = m["data"][name]
  else:
    mm = m["data"][m["modelname"] == name+".hddm"]
  
  for i in paralist:
    hddm.analyze.plot_posterior_nodes(mm.nodes_db.node[[str(i)+x for x in condition]])
    plt.xlabel(str(i))
    plt.ylabel('Posterior probability')
    if isinstance(name,int):
      plt.title('Posterior of' + m["modelname"][name] + 'group means')
    else:
      plt.title('Posterior of' + name +"'s " + str(i) + 'group means')
    plt.show()
    plt.savefig(str(i)+'.pdf')

def parallel(func, *args, show=False, thread=False, **kwargs):
  """
  并行计算
  :param func: 函数，必选参数
  :param args: list/tuple/iterable,1个或多个函数的动态参数，必选参数
  :param show:bool,默认False,是否显示计算进度
  :param thread:bool,默认False,是否为多线程
  :param kwargs:1个或多个函数的静态参数，key-word形式
  :return:list,与函数动态参数等长
  """
  import time
  from functools import partial
  from pathos.pools import ProcessPool, ThreadPool
  from tqdm import tqdm
  # 冻结静态参数
  p_func = partial(func, **kwargs)
  # 打开进程/线程池
  pool = ThreadPool() if thread else ProcessPool()
  try:
      if show:
          start = time.time()
          # imap方法
          with tqdm(total=len(args[0]), desc="计算进度") as t:  # 进度条设置
              r = []
              for i in pool.imap(p_func, *args):
                  r.append(i)
                  t.set_postfix({'并行函数': func.__name__, "计算花销": "%ds" % (time.time() - start)})
                  t.update()
      else:
          # map方法
          r = pool.map(p_func, *args)
      return r
  except Exception as e:
      print(e)
  finally:
      # 关闭池
      pool.close()  # close the pool to any new jobs
      pool.join()  # cleanup the closed worker processes
      pool.clear()  # Remove server with matching state

def gelman_rubin_test(df,times=5,**argm):
  """
  计算gelman_rubin r hat值, 默认samples=5000 burn=2000
  :param df:预处理后的数据
  :param times: chian的数量
  :return:gelman_rubin
  """
  import hddm
  from third_module import parallel
  data_sets = [df] * times
  def temp(df,**argm):
    import hddm
    m = hddm.HDDM(df,**argm)
    samples = 5000
    burn=2000
    m.find_starting_values()
    m.sample(samples,burn,dbname='gelman',db='pickle')
    return m
  ms = parallel(temp,data_sets,**argm)
  results = hddm.analyze.gelman_rubin(ms)
  return results

def params_table(m):
  import pandas as pd
  import numpy as np
  temp = m.gen_stats()
  tempf = lambda para:temp['mean'].iloc[temp.index.str.contains(para)].values
  stats = pd.DataFrame({
      "pars_id":np.append(["mean","std"],pd.unique(df_pars.subj_idx)),
      "a_boundary":tempf("a"),
      "v_drift":tempf("v"),
      "t_nondecision":tempf("^t")
      })
  return stats

def eachp_params(m):
  import matplotlib.pyplot as plt
  stats = m.gen_stats()
  fig,ax = plt.subplots()
  def tempf(para):
    ax.plot(stats.pars_id,stats[para],label=para)

  tempf("a_boundary")
  tempf("v_drift")
  tempf("t_nondecision")
  ax.set_title("parameters for each participants")
  ax.set_xlabel("group & each participants indexes")
  ax.legend()

def run_mult_chain(df, chains = 4, stim = False, **depends_on):
  """
  Run multiple chains

  df: df is a pandas dataframe;
  stim: stim is boolean
  *depends_on: depends_on is dict, note it cant not be a list.

 example:  
    ms = run_mult_chain(df)
    ms = run_mult_chain(df, depends_on={'v':'conf'}) 
  """

  from third_module import parallel
  from third_module import run_model
  dataset = [df] * chains

  if depends_on:
    ms = parallel(run_model, dataset, stim=stim, **depends_on)
  else:
    ms = parallel(run_model, dataset, stim=stim)
  
  return ms

def run_model(df, stim=False, **depends_on):
  """
   run model at one time
   df is pandas dataframe
   depends_on is experimental conditions, must be dict
   stim means that whether need to coding stimuli

   example:  
    ms = run_model(df)
    ms = run_model(df, depends_on={'v':'conf'})
  """

  import time
  import hddm

  samplenum= 5000
  burn = 2000
  p_outlier = 0.05
  postfix = ".hddm"

  starttime = time.time()
  if stim:
    m = hddm.HDDMStimCoding(df, p_outlier=p_outlier)
    name1 = "stim"
    if depends_on:
      m = hddm.HDDMStimCoding(df, p_outlier=p_outlier,depends_on=depends_on["depends_on"])
      name1 = "stim_cond"
  else:
    if depends_on:
      m = hddm.HDDM(df, p_outlier=p_outlier, depends_on=depends_on["depends_on"])
      name1 = "cond"
    else:
      m = hddm.HDDM(df, p_outlier=p_outlier)
      name1 = "Base"

  m.find_starting_values()
  m.sample(samplenum,burn,dbname=name1+'traces.db', db='pickle')
  m.save(name1+postfix)
  timegap = time.time()-starttime
  print(name1 + 'usage time: %.3f min' %(timegap/60))
  return m

def run_optimize(df,opt_methods='chisquare',p_outlier=0.05,**kwargs):

  from third_module import parallel
  import time

  def temp(df,opt_methods,p_outlier,**kwargs):
    import hddm
    if "depends_on" in kwargs:
      mm = hddm.HDDM(df,depends_on=kwargs["depends_on"],p_outlier=p_outlier)
    else:
      mm = hddm.HDDM(df,p_outlier=p_outlier)
    return mm.optimize(opt_methods)
  esd = [subj_data for _, subj_data in df.groupby('subj_idx')]
  esi = [i for i, _ in df.groupby('subj_idx')]

  st = time.time()
  params = parallel(temp,esd,opt_methods=opt_methods,p_outlier=p_outlier,**kwargs)
  print("cost",round((time.time()-st)/60),"minutes")

  return {'subj_idx':esi,'models':params}

def runms(funcs,paras):
  """ run multiple models with async processing
    It is necessary to rename sampler like "m.sample(5000,2000,dbname="name1",db="pickle")"

    para: funcs are hddm models
    para: paras are hddm name, data, conditons etc.
  """
  import multiprocessing as mp
  try:
    p = mp.Pool()
    multi_res = [p.apply_async(j[0],j[1]) for j in zip(funcs,paras)]
    rs = [res.get() for res in multi_res]
    return rs
  except Exception as e:
    print(e)

def run_model1(df_pars):
    import hddm
    m = hddm.HDDM(df_pars,p_outlier=0.05,depends_on={"v":"isfake","a":"isfake"})
    m.find_starting_values()
    m.sample(50,30,db='pickle')
    #m.save('db%i'%id)
    return m

def log_time(a_func):
    from functools import wraps
    import datetime
    @wraps(a_func) # 防止函数名被重写
    def wrapTheFunction(*args, **kwargs):
        start_t = datetime.datetime.now()
        aa = a_func(*args, **kwargs)
        end_t = datetime.datetime.now()
        elapsed_sec = (end_t - start_t).total_seconds()
        print("多进程计算 共消耗: " + "{:.2f}".format(elapsed_sec) + " 秒")
        return aa
    return wrapTheFunction
