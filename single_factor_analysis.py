import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
#一、数据导入————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# 读取Excel文件
sig_file = pd.ExcelFile("上证50指增sig.xlsx")
factor_file = pd.ExcelFile("上证50指增数据.xlsx")
ind_file = pd.ExcelFile("上证50行业.xlsx")
# 获取所有sheet的名称
sig_names = sig_file.sheet_names
factor_names = factor_file.sheet_names
# 现以通过字典访问不同的DataFrame
df_50sig = pd.read_excel(sig_file, sig_names[0], index_col='date')
df_stop_sig = pd.read_excel(sig_file, sig_names[1], index_col='date')
df_limit_sig = pd.read_excel(sig_file, sig_names[2], index_col='date')

df_ret = pd.read_excel(sig_file, sig_names[3], index_col='date')

df_free_value = pd.read_excel(factor_file, factor_names[0], index_col='date')
df_pb = pd.read_excel(factor_file, factor_names[1], index_col='date')
df_turnover_rate = pd.read_excel(factor_file, factor_names[2], index_col='date')
df_mom = pd.read_excel(factor_file, factor_names[3], index_col='date')
df_std = pd.read_excel(factor_file, factor_names[4], index_col='date')
df_roe = pd.read_excel(factor_file, factor_names[5], index_col='date')
df_beta = pd.read_excel(factor_file, factor_names[6], index_col='date')
df_dy = pd.read_excel(factor_file, factor_names[7], index_col='date')
#市值取对数
df_free_value=np.log(df_free_value)
df_free_value.replace([np.inf, -np.inf,np.nan], 0, inplace=True)
#读取行业
ind = pd.read_excel(ind_file, ind_file.sheet_names[0], index_col='date')
ind_names = pd.read_excel(ind_file, ind_file.sheet_names[1])



#二、数据初步筛选————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#选择因子并按一定要求筛选满足特定条件的
df_factor=(df_std) * df_50sig * df_stop_sig * df_limit_sig
df_ret=df_ret.shift(-1).fillna(0)#ret提前一阶，方便计算
# 筛选之后将所有的0值替换为空值，以此剔除极端值（需要留意有效原本因子值就是0的也会被剔除，特定问题特定分析）
df_factor.replace(0, np.nan, inplace=True)



#三、极端值剔除————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————N倍MAD剔除法（如果是全市场选股的话，这个剔除要放在筛选的前面）——————————————————————————————————————————————————
# 计算每行的均值和标准差
row_med = df_factor.median(axis=1)
row_mad = 1.4826 * abs(df_factor-row_med.to_numpy().reshape(-1, 1)).median(axis=1)
# 计算剔除条件
lower_bound = row_med - 3 * row_mad
upper_bound = row_med + 3 * row_mad
lower_bound =  lower_bound.to_numpy().reshape(-1, 1)
upper_bound =  upper_bound.to_numpy().reshape(-1, 1)
# 将超出范围的值设置为 NaN
df_factor[(df_factor < lower_bound) | (df_factor > upper_bound)] = np.nan
#————————————————————————————————————————————N倍标准差剔除法（如果是全市场选股的话，这个剔除要放在筛选的前面）——————————————————————————————————————————————————
# 计算每行的均值和标准差
#row_mean = df_factor.mean(axis=1)
#row_std = df_factor.std(axis=1)
# 计算剔除条件
#lower_bound = row_mean - 3 * row_std
#upper_bound = row_mean + 3 * row_std
#lower_bound =  lower_bound.to_numpy().reshape(-1, 1)
#upper_bound =  upper_bound.to_numpy().reshape(-1, 1)
# 将超出范围的值设置为 NaN
#df_factor[(df_factor < lower_bound) | (df_factor > upper_bound)] = np.nan
#————————————————————————————————————————————分位数剔除极值（如果是全市场选股的话，这个剔除要放在筛选的前面）——————————————————————————————————————————————————
#lower_bound = df_factor.quantile(0.01, axis=1)
#upper_bound  = df_factor.quantile(0.99, axis=1)
#lower_bound =  lower_bound.to_numpy().reshape(-1, 1)
#upper_bound =  upper_bound.to_numpy().reshape(-1, 1)
# 将超出范围的值设置为 NaN
#df_factor[(df_factor < lower_bound) | (df_factor > upper_bound)] = np.nan



#四、市值、行业中性化————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# 将每一行的非空元素合并成一个集合
df_inds=df_factor.notna()*ind
row_sets = df_inds.apply(lambda row: set(filter(lambda x: x != '', row)), axis=1)
# 找到所有行共有的字符
common_inds = list(set.intersection(*row_sets))
ind_names=ind_names[ind_names!=common_inds[1]].dropna().reset_index(drop=True)#这里有个伏笔，就是可能这个指数从来没有共有的行业，这种一般放弃该指数吧
inds=[]
for i in range(len(ind_names)):
    ind_name=ind_names.loc[i].values[0]
    inds.append((ind==ind_name)*1)
#正交化自定义函数
def qr_and_return(*args):
    matrix = pd.concat((args),axis=1)
    matrix.dropna(axis=0,inplace=True)
    matrix = matrix.loc[:, ~(matrix == 0).all()]#这一步是删掉列元素全为0的，也就是删掉股票都不属于的行业。此外行为0的不删，行为0的正好就是剔除的那个行业
    Q, R = np.linalg.qr(matrix)
    i=Q.shape[1]-1
    beta = pd.DataFrame(Q[:,i]*R[i,i], columns=[matrix.columns[i]], index=matrix.index)
    return beta
# x0截距项、中间是行业因子(inds存储)、x1市值因子、x2是要检验的因子
x0=df_factor.notna().astype(int)*1
x0.replace(0, np.nan,inplace=True)
x1=df_factor.notna().astype(int) * df_free_value
x1.replace(0, np.nan,inplace=True)
x2=df_factor.copy()
x_list= [df * x0 for df in inds[:]]+[x1]+[x2]
result = x0.apply(lambda row: qr_and_return(row, *[x.loc[row.name] for x in x_list]), axis=1)
df_factor = (pd.concat(result.tolist(),axis=1, ignore_index=False).T)*x0
df_factor = df_factor.round(5) #必须要加，因为正交化以后有很多接近0的数，比如1.1*e-15，6.5*e-17，这些本身应该是0，如果不约成0，会对分组产生很大误差



#五、因子标准化————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————Z_score标准化——————————————————————————————————————————————————————————————————————————
#df_factor = df_factor.apply(lambda x: (x-x.mean())/(x.std()), axis=1)
#————————————————————————————————————————————————0-1标准化——————————————————————————————————————————————————————————————————————————
df_factor = df_factor.apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=1)

file_path = "单因子集.xlsx"

# # 使用 ExcelWriter 写入文件
# with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
#     # 假设你想将数据框 df 写入新的 sheet，名称为 '新Sheet'
#     df_factor.to_excel(writer, sheet_name='股息率', index=True)  # 设置 index=True

    
    
#六、分组测试————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# 计算每一行的剔除nan以后，从大到小分5组，每组的分界线（排序等分法）
g=3
g_p = [i / g for i in range(1, g)]
thresholds_factor = df_factor.apply(lambda row: list(row.dropna().nlargest(int(len(row.dropna()) * p)).min() for p in g_p), axis=1)
# 计算每一行的剔除nan以后，从大到小分5组，每组的分界线（分位数等分法，可以用，但是不常用，相当于在分组的基础上引入了因子本身）
#thresholds_factor = df_factor.apply(lambda row: list(row.dropna().quantile([0.2, 0.4, 0.6, 0.8])), axis=1)
# 根据阈值进行标记
factor_group = df_factor.apply(lambda row: row.apply(lambda x: sum(x >= t for t in thresholds_factor[row.name]) + 1 if pd.notnull(x) else np.nan), axis=1)
#分组计算收益率和净值
#ret为了方便要跟着df_factor来剔除
ret_factor=df_factor.notna().astype(int)*df_ret
ret_factor.replace(0, np.nan, inplace=True)
ret = []
jz= []

#每一组组内等权
for i in range(1, g+1):
    ret_i = (factor_group == i) * ret_factor
    ret_i.replace(0, np.nan, inplace=True)
    ret_i=ret_i.mean(axis=1)
    ret_i.fillna(0,inplace=True)
    ret.append(ret_i)
    # 将每个收益率加1，得到增长率
    growth_rates = 1 + ret_i
    jz_i= np.cumprod(growth_rates)
    jz.append(jz_i)
    
    
    
#七、评价指标计算————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#计算IC\RANKIC
ret_rank=ret_factor.rank(axis=1)
factor_rank=df_factor.rank(axis=1)
ic=df_factor.corrwith(ret_factor,axis=1)
rankic=factor_rank.corrwith(ret_rank,axis=1)
icir=ic/(ic.rolling(12).std())#滚动12个月的IC波动率
rankicir=rankic/(rankic.rolling(12).std())
ic_cumsum=ic.cumsum()
icir_cumsum=icir.cumsum()
rankic_cumsum=rankic.cumsum()
rankicir_cumsum=rankicir.cumsum()
# ——————————————————————————————————— IC\RANKIC画图 ———————————————————————————————————
def plot_dual_axis_bar_line(df1, df2, ylabel1, ylabel2, title):
    fig, ax1 = plt.subplots(figsize=(8, 6))
    # 绘制第一列的主坐标轴条形图
    ax1.plot(df1.index.strftime('%Y-%m-%d'), df1, marker='o', linestyle='-', color='b', markersize=8)
    ax1.set_xlabel('Date')
    ax1.set_ylabel(ylabel1, color='b')
    # 创建第二个坐标轴（次坐标轴）
    ax2 = ax1.twinx()
    # 绘制第二列的次坐标轴折线图
    ax2.plot(df2.index.strftime('%Y-%m-%d'), df2, marker='o', linestyle='-', color='r', markersize=8)
    ax2.set_ylabel(ylabel2, color='r')
    # 添加标题
    plt.title(title)
    # 调整布局，避免重叠
    fig.tight_layout()
    # 显示图形
    plt.show()

# 绘制第一组数据
plot_dual_axis_bar_line(ic_cumsum, icir_cumsum, 'IC-cumsum', 'ICIR-cumsum', 'IC and ICIR-cumsum')
# 绘制第二组数据
plot_dual_axis_bar_line(rankic_cumsum, rankicir_cumsum, 'Rank IC-cumsum', 'Rank ICIR-cumsum', 'Rank IC and Rank ICIR-cumsum')
# ————————————————————————————————————净值画图————————————————————————————————————————————————————————
plt.figure(figsize=(10, 6))
# 遍历列表中的每个Series，并绘制成折线图
g=0
for jz_i in jz:
    g=g+1
    plt.plot(jz_i.index, jz_i.values, marker='o', linestyle='-', label=f'G{(g)}')
plt.plot(jz[(g-1)]/jz[0],label='max/min')
# 添加标题和标签
plt.xlabel('date')
plt.ylabel('net_value')
# 添加图例
plt.legend()
#plt.legend(['G1', 'G2', 'G3', 'G4', 'G5'])
# 显示图形
plt.show()
#plt.plot(jz[(g-1)]/jz[0])