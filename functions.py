
from scipy.integrate import solve_ivp
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# SIR model equations
def instant_vacc_SIR(t, y, phi, theta, betaw, gammaw, betam, gammam, epsilonL):
    """
    Modified SIR model for instant vaccination.
    """
    S, Iw, Rw, N, P, Im, Rm, CIw, CIm, = y

    dSdt = -(betaw*Iw+ betam*Im) * S #susceptible #note no v(t) because its incorporated into the 'seeding'
    dIwdt = betaw * Iw* (S + N + (1-epsilonL)*P) - gammaw * Iw #infections, wildtype
    dRwdt = gammaw * Iw - betam*Im*phi*Rw #recovered, wildtype

    dNdt = -(betam*Im + betaw*Iw) * N #no protection from vaccine
    dPdt =  -betaw*Iw*(1-epsilonL)*P - betam*Im*(1-((1-theta)*epsilonL))*P #some level of protection from vaccine

    dImdt = betam*Im*(S + N + phi*Rw + (1-((1-theta)*epsilonL))*P) - gammam * Im #infected mutants
    dRmdt = gammam * Im #recovered mutants

    dCIwdt = betaw * Iw* (S + N + (1-epsilonL)*P) #cumulative wildtype infections
    dCImdt = betam*Im*(S + N + phi*Rw + (1-((1-theta)*epsilonL))*P) #cumulative mutant infections


    return [dSdt, dIwdt, dRwdt, dNdt, dPdt, dImdt, dRmdt, dCIwdt, dCImdt]


#OUTCOME FINDING FUNCTIONS

def find_finalCIw(solution_y):
  S, Iw, Rw, N, P, Im, Rm, CIw, CIm,  = solution_y
  return(CIw[-1])

def find_finalCIm(solution_y):
  S, Iw, Rw, N, P, Im, Rm, CIw, CIm,  = solution_y
  return(CIm[-1])

def find_VI(solution_y, solution_nv):
  """
  Find  vaccine impact
  """

  CIw = find_finalCIw(solution_y) + find_finalCIm(solution_y)
  CIw_nv = find_finalCIw(solution_nv) + find_finalCIm(solution_nv)

  return ((CIw_nv - CIw) / CIw_nv) * 100


def return_nv_y0(y0):
  """
  Returns version of y0 without vaccinations.
  y0 in order of [S0, Iw0, Rw0, N0, P0, Im0, Rm0, CIw0, CIm0]
  """
  new_y0 = y0.copy()
  S0, Iw0, Rw0, N0, P0, Im0, Rm0, CIw0, CIm0 = new_y0

  new_y0 = [S0+N0+P0, Iw0, Rw0, 0, 0, Im0, Rm0, CIw0, CIm0]

  return new_y0


def calc_Rts(S, N, P, Rw):
  """
  Calculate effective reproduction number at each t.
  """
  Rtw = []
  Rtm = []

  for i in range(len(S)):
    Rtw.append(Rwnaught*(S[i]+N[i]+(P[i]*(1-epsilonL))))
    Rtm.append(Rmnaught*(S[i]+N[i]+(P[i]*(1-(1-theta)*epsilonL))+phi*Rw[i]))

  return(Rtw, Rtm)


def find_deltaCI(solution_aon, solution_leaky, recovered_threshold = None, var = False):
  """
  Calculate difference in total cumulative infections between aon and leaky solutions
  """
  if var == True:
    if check_threshold_modes(solution_aon, solution_leaky, recovered_threshold) == True:
      return np.nan

  CI_aon = find_finalCIw(solution_aon) + find_finalCIm(solution_aon)
  CI_leaky = find_finalCIw(solution_leaky) + find_finalCIm(solution_leaky)

  return CI_leaky - CI_aon


def find_deltaVI(solution_aon, solution_leaky, solution_nv, recovered_threshold = None, var = False):
  """
  Find difference in vaccine impact
  """
  if recovered_threshold != None and var == True:
    if check_threshold_modes(solution_aon, solution_leaky, recovered_threshold) == True:
      return np.nan

  VI_aon = find_VI(solution_aon, solution_nv)
  VI_leaky = find_VI(solution_leaky, solution_nv)

  return VI_aon - VI_leaky

def compare_mutantimpact(solution_aon, solution_leaky, solution_aon_nm,
                         solution_leaky_nm, recovered_threshold = None, var = False):
    """
    Calculate the difference in "proportional mutant impact" as a ratio of proportions.
    """
    if var == True:
      if check_threshold_modes(solution_aon, solution_leaky, recovered_threshold) == True:
        return np.nan

    CI_aon_m = find_finalCIw(solution_aon) + find_finalCIm(solution_aon)
    CI_leaky_m = find_finalCIw(solution_leaky) + find_finalCIm(solution_leaky)

    CI_aon_nm = find_finalCIw(solution_aon_nm) + find_finalCIm(solution_aon_nm)
    CI_leaky_nm = find_finalCIw(solution_leaky_nm) + find_finalCIm(solution_leaky_nm)

    mutant_impact_aon = CI_aon_m/CI_aon_nm
    mutant_impact_leaky = CI_leaky_m/CI_leaky_nm


    return mutant_impact_aon/mutant_impact_leaky


#function that takes aon and leaky
#checks if final ciw has above threshold for both
#if one has and the other doesn't, send true or smth
#break loops / early return in other functions (that calc measures)

def check_threshold_modes(solution_aon, solution_leaky, recovered_threshold):

  #if aon triggers seeding but leaky doesnt
  if (find_finalCIw(solution_aon) > recovered_threshold) and (find_finalCIw(solution_leaky) < recovered_threshold):
    return True

  #vice versa
  if find_finalCIw(solution_leaky) > recovered_threshold and  (find_finalCIw(solution_aon) < recovered_threshold):
    return True

  return False

#INDIVIDUAL TIME COURSE PLOTTING FUNCTIONS

def plot_basic(solution_t, solution_y, ax=None, title="SIR sim w variant",
               ylim_max=1, includeRt=False, figsize=(10,6), lw=1.5, fontsize = 14, legend = False):
  """
  Plot basic time series. Ax not needed, will plot on its own.
  """
  if ax is None:
      fig, ax = plt.subplots(figsize=(figsize))

  t_full = solution_t
  S, Iw, Rw, N, P, Im, Rm, CIw, CIm,  = solution_y

  ax.plot(t_full, S, label='Susceptible', lw=lw)
  ax.plot(t_full, Iw, label='Infected (wildtype)', color="#d768d1", lw=lw)
  ax.plot(t_full, Im, label='Infected (mutant)', color="#fc9432", lw=lw)
  ax.plot(t_full, Rw, label='Recovered (wildtype)', color="#d768d1", linestyle="dashed", lw=lw)
  ax.plot(t_full, Rm, label='Recovered (mutant)', color="#fc9432", linestyle="dashed", lw=lw)
  ax.plot(t_full, N, label='Vaccinated, none', color = "C4", lw=lw)
  ax.plot(t_full, P, label='Vaccinated, no/partial/perfect', color = "C2", lw=lw)

  if includeRt:
      Rtw, Rtm = calc_Rts(S, N, P, Rw)
      ax.plot(t_full, Rtw, label='Rt of wildtype', linestyle='dashed', lw=lw)
      ax.plot(t_full, Rtm, label='Rt of mutant', linestyle='dashed', lw=lw)

  ax.set_xlabel('Time (days)', fontsize=fontsize)
  ax.set_ylabel('Proportion of Population', fontsize=fontsize)
  ax.set_ylim(0, ylim_max)
  if legend == True:
    ax.legend()
  ax.grid(True)
  ax.set_title(title, fontsize = fontsize)

  return ax

def plot_infecteds(solution_t, solution_y, ax=None, title="two_strain, infecteds only",
                   ylim_max = 0.2, fontsize = 14):
  """
  Plot time series of Iw and Im only (infections). Ax not needed, will plot on its own.
  """
  if ax is None:
      fig, ax = plt.subplots(figsize = (10,6))

  t_full = solution_t
  S, Iw, Rw, N, P, Im, Rm, CIw, CIm,  = solution_y

  ax.plot(t_full, Iw, label='Infected (wildtype)', color = "#d768d1")
  ax.plot(t_full, Im, label='Infected (mutant)',  color = "#fc9432")

  ax.set_xlabel('Time (days)', fontsize = fontsize)
  ax.set_ylabel('Proportion of Population', fontsize = fontsize)
  ax.set_ylim(0, ylim_max)
  ax.legend()
  ax.grid(True)
  ax.set_title(title, fontsize = fontsize)

  return ax

def plot_cums(solution_t, solution_y, ax=None, title="two_strain, infecteds only",
              ylim_max = 1, lw = 4, fontsize = 14, figsize = (10,6)):
  """
  Plot time courses of CIw and CIm only (cumulative infections). Ax not needed, will plot on its own.
  """
  if ax is None:
      fig, ax = plt.subplots(figsize = figsize)

  t_full = solution_t
  S, Iw, Rw, N, P, Im, Rm, CIw, CIm,  = solution_y

  ax.plot(t_full, CIw, label='Infected (wildtype)',  color = "#d768d1", lw = lw)
  ax.plot(t_full, CIm, label='Infected (mutant)', color = "#fc9432", lw = lw)

  ax.set_xlabel('Time (days)', fontsize = fontsize)
  ax.set_ylabel('Proportion of Population', fontsize = fontsize)
  ax.set_title(title, fontsize = fontsize)
  ax.set_ylim(0, ylim_max)
  ax.grid(True)

  return ax

#PLOTTERS OF MULTIPLE SOLUTIONS

def plot_modes_all(solutions_list: list, param: str, param_list: list, ylim_max = 1,
                  suptitle = None, epsilons: list = None):
  """
  Plots time series, all curves, for all modes given in solutions_list.
  Solutions list is nested list with lists of aon, leaky, and inter. All lists of modes in that order.
  Epsilons is an optional list of tuples that specify epsA and epsL values, for the purpose of labelling.
  """

  #param is the param that is varied, list is the varied values
  fig, axes = plt.subplots(len(solutions_list), len(param_list), figsize=(20, 20), sharex=True, sharey=True)

  for j in range(len(solutions_list)):
    #replace this w rows later
    if j == 0:
      mode = "aon"
    if j == 1:
      mode = "leaky"
    if j == 2:
      mode = "intermediate"

    for i in range(len(param_list)):
      solution = solutions_list[j][i]
      param_i = param_list[i]
      title = f"{param} = {param_i}, {mode}"

      if epsilons != None:
        epsilon_tuple = epsilons[j]
        title = f"{param} = {param_i}, {mode},\n (epsA, epsL) = {epsilon_tuple}"

      plot_basic(solution[0], solution[1], title = title,
                 ax = axes[j][i], ylim_max = ylim_max)

  if suptitle != None:
    fig.suptitle(suptitle)


def plot_modes_inf(solutions_list: list, param: str, param_list: list, ylim_max = 0.3,
                   suptitle = None, epsilons: list = None):

  """
  Plots time series, all curves, for all modes given in solutions_list.
  Solutions list is nested list with lists of aon, leaky, and inter. All lists of modes in that order.
  Epsilons is an optional list of tuples that specify epsA and epsL values, for the purpose of labelling.
  """

  #param is the param that is varied, lsit is the varies values
  fig, axes = plt.subplots(len(solutions_list), len(param_list), figsize=(20, 20), sharex=True, sharey=True)

  for j in range(len(solutions_list)):
    #replace this w rows later
    if j == 0:
      mode = "aon"
    if j == 1:
      mode = "leaky"
    if j == 2:
      mode = "intermediate"

    for i in range(len(param_list)):
      solution = solutions_list[j][i]
      param_i = param_list[i]
      title = f"{param} = {param_i}, {mode}"

      if epsilons != None:
        epsilon_tuple = epsilons[j]
        title = f"{param} = {param_i}, {mode},\n (epsA, epsL) = {epsilon_tuple}"

      plot_infecteds(solution[0], solution[1], title = title,
                 ax = axes[j][i], ylim_max = ylim_max)
  if suptitle != None:
      fig.suptitle(suptitle)


def plot_modes_cum(solutions_list: list, param: str, param_list: list, ylim_max = 1.0,
                   suptitle = None, epsilons: list = None):
  """
  Plots time series, only cumulative, for all modes given in solutions_list.
  Solutions list is nested list with lists of aon, leaky, and inter. All lists of modes in that order.
  Epsilons is an optional list of tuples that specify epsA and epsL values, for the purpose of labelling.
  """

  #param is the param that is varied, lsit is the varies values
  fig, axes = plt.subplots(3, len(param_list), figsize=(20, 20), sharex=True, sharey=True)

  for j in range(len(solutions_list)):
    #replace this w rows later
    if j == 0:
      mode = "aon"
    if j == 1:
      mode = "leaky"
    if j == 2:
      mode = "intermediate"

    for i in range(len(param_list)):
      solution = solutions_list[j][i]
      param_i = param_list[i]
      title = f"{param} = {param_i}, {mode}"


      if epsilons != None:
        epsilon_tuple = epsilons[j]
        title = f"{param} = {param_i}, {mode},\n (epsA, epsL) = {epsilon_tuple}"

      plot_cums(solution[0], solution[1], title = title,
                 ax = axes[j][i], ylim_max = ylim_max)
  if suptitle != None:
      fig.suptitle(suptitle)

#BAR PLOTS OF MUTANT IMPACT MEASURES AND CIs

def plot_mutprop(df, param_varied: str, ax = None, title = "Mut. Proportion of Infections", ylim_max = 1, puor = False):
  """
  Plots mutant proportion of infections using dataframe that includes:
  1) whether mutant exists
  2) CIs for mutant and WT
  3) mode
  """

  if ax is None:
        fig, ax = plt.subplots(figsize = (10,6))


  new_df = df[df['mutant exists?']==True].copy()
  new_df["Mutant proportion of infections"] = new_df["Cumulative Infections mutant"]/(new_df["Cumulative Infections WT"]+new_df["Cumulative Infections mutant"])

  pivot_df = new_df.pivot(index = param_varied,
                          columns = "mode",
                          values = "Mutant proportion of infections")

  #plot
  puor_colors = ["#fbbf72", "#481a72",]

  if puor == True:
    pivot_df.plot.bar(ax=ax, rot=0, color=puor_colors)
  else:
    pivot_df.plot.bar(ax=ax, rot=0)

  ax.set_title(title)
  ax.set_ylabel("Proportion of Infections")
  ax.set_xlabel(param_varied)
  ax.set_ylim(0, ylim_max)
  ax.grid(alpha = 0.3)

  return ax

def plot_mutantimpact(df, param_varied: str, ax = None,
                      title = "Prop. Increase in Cumulative Infections\nrelative to no-mutant counterfactual",
                      ylim_max = 5, ylim_min = 0, puor = False):
  """
  Plots mutant impact, i.e. proportional increase of infections relative to no mutant counterpart.
  Requires no-mutant counterfactual to exist in df.
  Data frame includes:
  1) whether mutant exists
  2) CIs for mutant and WT
  3) mode
  For mutant and no mutant scenarios.
  """
  if ax == None:
    fig, ax = plt.subplots(figsize = (10,6))

  #separate cum inf data
  df_mut = df[df["mutant exists?"] == True].copy()
  df_nomut = df[df["mutant exists?"] == False].copy()

  #wrangle to merge by mode/VC
  df_mut["Total_with_mut"] = (df_mut["Cumulative Infections WT"] +
                              df_mut["Cumulative Infections mutant"])

  merged = df_mut.merge(df_nomut[[param_varied, "mode", "Cumulative Infections WT"]],
                        on=[param_varied, "mode"],
                        suffixes=("_mut", "_nomut"))

  merged["mutant impact"] = (merged["Total_with_mut"] /
                             merged["Cumulative Infections WT_nomut"])

  impact_df = merged[[param_varied, "mode", "mutant impact"]]

  #pivot for bar plot
  pivot_df = impact_df.pivot(index = param_varied,
                             columns = "mode",
                             values = "mutant impact")

  puor_colors = ["#fbbf72", "#481a72"]
  if puor == True:
    pivot_df.plot.bar(ax=ax, rot=0, color=puor_colors)

  else:
    pivot_df.plot.bar(ax=ax, rot=0)

  ax.set_title(title)
  ax.set_xlabel(param_varied)
  ax.set_ylim(ylim_min, ylim_max)
  ax.grid(alpha = 0.3)

  return ax

def plot_CIs(df, param_varied: str, suptitle, nm_cf = False,
             figsize=(15, 5), ylim_max = 1.8, ):

  """
  Plots cumulative infections as bar plots for wildtype only, mutant only, their sum, and no mutant version (if boolean nm_cf = True).
  For all-or-nothing, leaky, and intermediate modes.
  """

  #if counterfactual needs to be plotted
  if nm_cf == True:
    nm_df = df[df["mutant exists?"] == False] #no mutant df
    df = df[df["mutant exists?"] == True]

    # pivot long to wide, non-mutant
    pivot_nm = nm_df.pivot(index=(param_varied),
                           columns="mode",
                           values="Cumulative Infections WT")
    print(pivot_nm)
    #figure w 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=figsize,sharex=True, sharey=True)

  else: #if no mutant cf does not exist,
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True)

  pivot_WT = df.pivot(index=(param_varied),
                      columns="mode",
                      values="Cumulative Infections WT")

  pivot_mut = df.pivot(index=(param_varied),
                        columns="mode",
                        values="Cumulative Infections mutant")

  total_df = pivot_WT + pivot_mut

  #creates axes and plots
  pivot_WT.plot.bar(ax=axes[0], rot=0)
  axes[0].set_title("WT Cumulative Infections")
  axes[0].set_ylabel("Cumulative Infections")
  axes[0].set_xlabel(param_varied)
  axes[0].grid(True, alpha = 0.3)

  pivot_mut.plot.bar(ax=axes[1], rot=0)
  axes[1].set_title("Mutant Cumulative Infections")
  axes[1].set_xlabel(param_varied)
  axes[1].grid(True, alpha = 0.3)

  total_df.plot.bar(ax=axes[2], rot=0)
  axes[2].set_title("Total (WT + Mutant)")
  axes[2].set_xlabel(param_varied)
  axes[2].set_ylim(0, ylim_max)
  axes[2].grid(True, alpha = 0.3)

  if nm_cf == True:
    pivot_nm.plot.bar(ax=axes[3], rot=0)
    axes[3].set_title("WT Cumulative Infections,\nno-mutant counterfactual ")
    axes[3].set_xlabel(param_varied)
    axes[3].grid(True, alpha = 0.3)

  fig.suptitle(suptitle)

  plt.tight_layout()
  plt.show()



def plot_CIs_2(df, param_varied: str, suptitle, nm_cf = False,
             figsize=(15, 5), ylim_max = 1.8, ):
  """
  Plots cumulative infections as bar plots for total infections and no mutant version (if boolean nm_cf = True). No wildtype only/mutant only.
  For all-or-nothing, leaky, and intermediate modes.
  """

  #if counterfactual needs to be plotted
  if nm_cf == True:
    nm_df = df[df["mutant exists?"]==False] #no mutant df
    df = df[df["mutant exists?"]==True]

    # pivot long to wide, non-mutant
    pivot_nm = nm_df.pivot(index=(param_varied),
                    columns="mode",
                    values="Cumulative Infections WT")

    #figure w 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)

  else: #if nm cf does not need to be plotted
    fig, axes = plt.subplots(1, 1, figsize=figsize, sharex=True, sharey=True)
    axes = [axes]

  pivot_WT = df.pivot(index=(param_varied),
                      columns="mode",
                      values="Cumulative Infections WT")

  pivot_mut = df.pivot(index=(param_varied),
                        columns="mode",
                        values="Cumulative Infections mutant")

  total_df = pivot_WT + pivot_mut

  total_df.plot.bar(ax=axes[0], rot=0,)
  axes[0].set_title("Total (WT + Mutant)")
  axes[0].set_xlabel(param_varied)
  axes[0].set_ylim(0, ylim_max)
  axes[0].grid(True, alpha=0.3)

  if nm_cf == True:
      pivot_nm.plot.bar(ax=axes[1], rot=0,)
      axes[1].set_title("WT Cumulative Infections,\nno-mutant counterfactual")
      axes[1].set_xlabel(param_varied)
      axes[1].grid(True, alpha=0.3)

  #legend to bottom

  return fig


def plot_CIs_2(df, param_varied: str, suptitle, nm_cf = False,
             figsize=(15, 5), ylim_max = 1.8, ):
  """
  Plots cumulative infections as bar plots for total infections and no mutant version (if boolean nm_cf = True). No wildtype only/mutant only.
  For all-or-nothing, leaky, and intermediate modes.
  """

  nm_df = df[df["mutant exists?"]==False]
  df = df[df["mutant exists?"]==True]

  pivot_nm = nm_df.pivot(index=(param_varied),
                  columns="mode",
                  values="Cumulative Infections WT")

  fig, axes = plt.subplots(1, 2, figsize=figsize,)

  pivot_WT = df.pivot(index=(param_varied),
                      columns="mode",
                      values="Cumulative Infections WT")

  pivot_mut = df.pivot(index=(param_varied),
                        columns="mode",
                        values="Cumulative Infections mutant")

  total_df = pivot_WT + pivot_mut

  total_df.plot.bar(ax=axes[0], rot=0, legend=False)
  axes[0].set_title("Total (WT + Mutant)")
  axes[0].set_xlabel(param_varied)
  axes[0].set_ylim(0, ylim_max)
  axes[0].grid(True, alpha=0.3)

  pivot_nm.plot.bar(ax=axes[1], rot=0, legend=False)
  axes[1].set_title("WT Cumulative Infections,\nno-mutant counterfactual")
  axes[1].set_xlabel(param_varied)
  axes[1].set_ylim(0, ylim_max)
  axes[1].grid(True, alpha=0.3)

  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=len(labels))
  fig.suptitle(suptitle)

  return fig

def plot_VIs(df, param_varied: str, suptitle,
             figsize=(15, 5), ylim_max = 1.8):

  """
  Plots vaccine impact as bar plots for total  mutant version and no mutant version (if boolean nm_cf = True).
  For all-or-nothing, leaky, and intermediate modes.
  """

  nm_df = df[df["mutant exists?"] == False] #no mutant df
  df = df[df["mutant exists?"] == True]

  # pivot long to wide, non-mutant
  pivot_nm = nm_df.pivot(index=(param_varied),
                          columns="mode",
                          values="Vaccine Impact")


  fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)

  pivot_m = df.pivot(index=(param_varied),
                      columns="mode",
                      values="Vaccine Impact")


  #creates axes and plots
  pivot_m.plot.bar(ax=axes[0], rot=0)
  axes[0].set_title("Mutant")
  axes[0].set_ylabel("Cumulative Infections")
  axes[0].set_xlabel(param_varied)
  axes[0].grid(True, alpha = 0.3)

  pivot_nm.plot.bar(ax=axes[1], rot=0)
  axes[1].set_title("No mutant")
  axes[1].set_xlabel(param_varied)
  axes[1].grid(True, alpha = 0.3)

  fig.suptitle(suptitle)

  plt.tight_layout()
  plt.show()

#SOLVERS OF MULTIPLE SOLUTIONS OF DIFF MODES
def solve_aonleaky(VE, theta, phi, beta, gamma,
                   S0, Iw0, Im0, Rw0, Rm0, CIw0, CIm0, t, mutant = True):
  """
  Solves for aon and leaky modes. Uses global initial values and phi/theta params.
  Can be adjusted for both mutant and non-mutant scenarios by changing mutant boolean.
  """
  if mutant == False:
    Im_new = 0
    Iw_new = Iw0 + Im0
  else:
    Im_new = Im0
    Iw_new = Iw0

  # aon
  epsilona = VE
  epsilonL_aon = 1

  N0 = (1-epsilona)*(1-(S0+Iw_new+Rw0+Rm0+Im_new))
  P0 = 1-(S0+Iw_new+Rw0+Rm0+Im_new+N0)

  if Iw_new+S0+Rw0+N0+P0+Rm0+Im_new != 1:
      print(f"ERROR_ERROR_AON: {Iw_new+S0+Rw0+N0+P0+Rm0+Im_new}")
      print(f"Iw_new:{Iw_new}; Im_new:{Im_new} S0:{S0}; Rw0:{Rw0}; Rm0:{Rm0}; N0:{N0}; P0:{P0};")

  y0_aon = [S0, Iw_new, Rw0, N0, P0, Im_new, Rm0, CIw0, CIm0]

  sol_aon = solve_ivp(instant_vacc_SIR, [0, t], y0_aon,
                      args=(phi, theta, beta, gamma, beta, gamma, epsilonL_aon),
                      t_eval=np.linspace(0, t, t))


  # leaky
  epsilona = 1
  epsilonL_leaky = VE

  N0 = (1-epsilona)*(1-(S0+Iw_new+Rw0+Rm0+Im_new))
  P0 =  1-(S0+Iw_new+Rw0+Rm0+Im_new+N0)

  if Iw_new+S0+Rw0+N0+P0+Rm0+Im_new != 1:
      print(f"ERROR_ERROR_LEAKY: {Iw_new+S0+Rw0+N0+P0+Rm0+Im_new}")
      print(f"Iw_new:{Iw_new}; Im_new:{Im_new} S0:{S0}; Rw0:{Rw0}; Rm0:{Rm0}; N0:{N0}; P0:{P0};")

  y0_leaky = [S0, Iw_new, Rw0, N0, P0, Im_new, Rm0, CIw0, CIm0]

  sol_leaky = solve_ivp(instant_vacc_SIR, [0, t], y0_leaky,
                        args=(phi, theta, beta, gamma, beta, gamma, epsilonL_leaky),
                        t_eval=np.linspace(0, t, t))

  #return as list so seeded and nonseeded functionsare compatible
  y_aon = sol_aon.y
  t_aon = sol_aon.t

  y_leaky = sol_leaky.y
  t_leaky = sol_leaky.t

  return [t_aon, y_aon], [t_leaky, y_leaky]

def solve_aonleakyinter(VE, theta, phi, beta, gamma,
                        S0, Iw0, Im0, Rw0, Rm0, CIw0, CIm0, t, mutant = True):
  """
  Solves for aon leaky and intermediate modes. Uses global initial values and phi/theta params.
  Can be adjusted for both mutant and non-mutant scenarios by changing mutant boolean.
  """

  if mutant == False:
    Im_new = 0
    Iw_new = Iw0 + Im0
  else:
    Im_new = Im0
    Iw_new = Iw0

  # aon
  epsilona = VE
  epsilonL_aon = 1

  N0 = (1-epsilona)*(1-(S0+Iw_new+Rw0+Rm0+Im_new))
  P0 = 1-(S0+Iw_new+Rw0+Rm0+Im_new+N0)

  if Iw_new+S0+Rw0+N0+P0+Rm0+Im_new != 1:
    print(f"ERROR_ERROR_AON: {Iw_new+S0+Rw0+N0+P0+Rm0+Im_new}")
    print(f"Iw_new:{Iw_new}; Im_new:{Im_new} S0:{S0}; Rw0:{Rw0}; Rm0:{Rm0}; N0:{N0}; P0:{P0};")

  y0_aon = [S0, Iw_new, Rw0, N0, P0, Im_new, Rm0, CIw0, CIm0]

  sol_aon = solve_ivp(instant_vacc_SIR, [0, t], y0_aon,
                      args=(phi, theta, beta, gamma, beta, gamma, epsilonL_aon),
                      t_eval=np.linspace(0, t, t))


  # leaky
  epsilona = 1
  epsilonL_leaky = VE

  N0 = (1-epsilona)*(1-(S0+Iw_new+Rw0+Rm0+Im_new))
  P0 = 1-(S0+Iw_new+Rw0+Rm0+Im_new+N0)

  y0_leaky = [S0, Iw_new, Rw0, N0, P0, Im_new, Rm0, CIw0, CIm0]

  if Iw_new+S0+Rw0+N0+P0+Rm0+Im_new != 1:
    print(f"ERROR_ERROR_LEAKY: {Iw_new+S0+Rw0+N0+P0+Rm0+Im_new}")
    print(f"Iw_new:{Iw_new}; Im_new:{Im_new} S0:{S0}; Rw0:{Rw0}; Rm0:{Rm0}; N0:{N0}; P0:{P0};")

  sol_leaky = solve_ivp(instant_vacc_SIR, [0, t], y0_leaky,
                        args=(phi, theta, beta, gamma, beta, gamma, epsilonL_leaky),
                        t_eval=np.linspace(0, t, t))

  # inter
  epsilona = math.sqrt(VE)
  epsilonL_inter = math.sqrt(VE)

  N0 = (1-epsilona)*(1-(S0+Iw_new+Rw0+Rm0+Im_new))
  P0 = 1-(S0+Iw_new+Rw0+Rm0+Im_new+N0)

  if Iw_new+S0+Rw0+N0+P0+Rm0+Im_new != 1:
    print(f"ERROR_ERROR_INTER: {Iw_new+S0+Rw0+N0+P0+Rm0+Im_new}")
    print(f"Iw_new:{Iw_new}; Im_new:{Im_new} S0:{S0}; Rw0:{Rw0}; Rm0:{Rm0}; N0:{N0}; N0:{P0};")

  y0_inter = [S0, Iw_new, Rw0, N0, P0, Im_new, Rm0, CIw0, CIm0]

  sol_inter = solve_ivp(instant_vacc_SIR, [0, t], y0_inter,
                        args=(phi, theta, beta, gamma, beta, gamma, epsilonL_inter),
                        t_eval=np.linspace(0, t, t))

  #make these into nested lists so they're consistent with seeded returns
  y_aon = sol_aon.y
  t_aon = sol_aon.t

  y_leaky = sol_leaky.y
  t_leaky = sol_leaky.t

  y_inter = sol_inter.y
  t_inter = sol_inter.t

  return  [t_aon, y_aon], [t_leaky,y_leaky], [t_inter, y_inter]

def create_suptitle(gamma = None, VC=None, theta=None, VE=None, R0=None,
                    Im0=None, Iw0=None, recovered_threshold = None,
                    fraction_seed = None, suptitle=None, ):
  """
  Creates subtitle for plotting featuring initial values and parameters.
  """
  details = []
  if VC is not None: details.append(f"VC={VC}")
  if theta is not None: details.append(f"θ={theta}")
  if VE is not None: details.append(f"VE={VE}")
  if R0 is not None: details.append(f"R0={R0}")
  if Iw0 is not None: details.append(f"Iw0={Iw0}")
  if Im0 is not None: details.append(f"Im0={Im0}")
  if gamma is not None: details.append(f"gamma={gamma}")
  if recovered_threshold is not None: details.append(f"recovered_threshold={recovered_threshold}")
  if fraction_seed is not None: details.append(f"fraction_seed={fraction_seed}")
  detail_str = ", ".join(details)

  if suptitle:
      return f"{suptitle}\n{detail_str}"
  else:
      return detail_str


def solve_nv(theta, phi, beta, gamma, Iw0, Im0, Rw0, Rm0, t, nm_cf = False, var = False,
             recovered_threshold = None, fraction_seed = None):
  """
  Gives non-vaccinated solutions for mutant and non-mutant scenarios.
  Variant seeding scenario can still occur.

  Only needs infected and recovered, creates y0 based off those v alues.
  """
  #inital pops
  CIw0 = Iw0
  CIm0 = Im0

  y0_mut_nv = [1-Iw0-Im0, Iw0, Rw0, 0, 0, Im0, Rm0, CIw0, CIm0]

  if var == False:
    nv_mut_sol = solve_ivp(instant_vacc_SIR, [0, t], y0_mut_nv,
                           args=(phi, theta,beta, gamma, beta, gamma, 0),
                          t_eval=np.linspace(0, t, t))
    #make consistent w seeded versions
    nv_mut_sol = [nv_mut_sol.t, nv_mut_sol.y]

  #epsilon a and l should not be used, thus 999. not using 0 so if N and P arise, it will be clear something is wrong
  else:
    nv_mut_sol = solve_seeded_var(y0_mut_nv, t, phi, theta, beta, gamma, beta, gamma,
                                  999, 999, recovered_threshold, fraction_seed)


  #no mutant version means new infected wildtype = old infected mutant proportion + old wt
  #also means no seeding at all (no mut, no vacc)
  if nm_cf == True:
    y0_nm_nv = [1-Iw0-Im0, Iw0+Im0, Rw0, 0, 0, 0, Rm0, CIw0+CIm0, 0]

    nv_nm_sol = solve_ivp(instant_vacc_SIR, [0, t], y0_nm_nv,
                    args=(phi, theta,beta, gamma, beta, gamma, 0),
                    t_eval=np.linspace(0, t, t))

    nv_nm_sol = [nv_nm_sol.t, nv_nm_sol.y]
    return nv_mut_sol, nv_nm_sol

  else:
    return nv_mut_sol

#INDIVIDUAL SOLVERS FOR VACCINE SEEDING
def find_tv_vacc(y0, t, phi, theta, betaw, gammaw, betam, gammam, epsilonL, recovered_threshold):
  """
  Find time of vaccine seeding based on reach_size terminate event. I.e. solve_ivp runs until event.
  """
  # defined inside find_tv so it closes over recovered_threshold since i cant add more args in solve ivp
  def _reach_size(t, y, phi, theta, betaw, gammaw, betam, gammam, epsilonL):
      return (y[7]) - recovered_threshold #TOTAL INFECTIONS reaches certain amount
  _reach_size.terminate = True

  sol_findtv = solve_ivp(instant_vacc_SIR, [0, t], y0,
                         args=(phi, theta, betaw, gammaw, betam, gammam, epsilonL),
                         events=_reach_size, t_eval=np.linspace(0, t, t))

  #i.e. no event found, threshold not reached
  if len(sol_findtv.t_events[0]) == 0:
      return None
  #else, return time of threshold pass
  tv = np.ravel(np.array(sol_findtv.t_events))[0]
  return tv

def find_tv_var(y0, t, phi, theta, betaw, gammaw, betam, gammam, epsilonL, recovered_threshold):

  def _reach_size(t, y, phi, theta, betaw, gammaw, betam, gammam, epsilonL):
      return (y[7]) - recovered_threshold #recovered popn reaches certain amount
  _reach_size.terminate = True

  sol_findtv = solve_ivp(instant_vacc_SIR, [0,t], y0,
                  args=(phi, theta, betaw, gammaw, betam, gammam, epsilonL),
                  events=_reach_size,
                  t_eval = np.linspace(0, t, t)) #changed dense output to np.linspace

  if len(sol_findtv.t_events[0]) == 0:
      return None   # or np.nan

  # get tv and seed Iv
  tv = np.ravel(np.array(sol_findtv.t_events))[0]

  return tv

#solver for vaccine only
def solve_seeded_vacc(y0, t, phi, theta, betaw, gammaw, betam, gammam, epsilonL,
                      epsilona, recovered_threshold, fraction_seed):


  tv = find_tv_vacc(y0, t, phi, theta, betaw, gammaw, betam, gammam, epsilonL, recovered_threshold)
  print(f"tv = {tv}")

  if tv is None: #no seeding occurs, just solve regularly
    sol_full = solve_ivp(instant_vacc_SIR, [0, t], y0,
                         args=(phi, theta, betaw, gammaw, betam, gammam, epsilonL),
                         t_eval=np.linspace(0, t, t))
    y_full = sol_full.y
    t_full = sol_full.t

  else:
    #first part of soln, going until seeding time
    sol1 = solve_ivp(instant_vacc_SIR, [0, tv], y0,
                     args=(phi, theta, betaw, gammaw, betam, gammam, epsilonL),
                     t_eval=np.linspace(0, tv, t))
    #take y at time of seeding
    y_tv = sol1.y[:, -1]

    #check everything works correctly
    print(f"y_tv S = {y_tv[0]}")
    print(f"y_tv Rw = {y_tv[2]}")
    print(f"frac_seed_vacc = {fraction_seed}")

    #change y to reflect seeding
    last_S = (sol1.y[:, -1][0])
    y_tv[0] -= fraction_seed*last_S #take from susceptibles
    #remove susceptibles and move to N and P
    y_tv[3] += (1-epsilona) * (fraction_seed*last_S)
    y_tv[4] += epsilona * (fraction_seed*last_S)
    
    print(y_tv[3], y_tv[4])

    #second part of solution
    sol2 = solve_ivp(instant_vacc_SIR, [tv, t], y_tv,
                     args=(phi, theta, betaw, gammaw, betam, gammam, epsilonL),
                     t_eval=np.linspace(tv, t, t))

    #combine two solutions into full soln
    t_full = np.concatenate([sol1.t, sol2.t])
    y_full = np.hstack([sol1.y, sol2.y])

  return [t_full, y_full]

def solve_seeded_var(y0, t, phi, theta, betaw, gammaw, betam, gammam, epsilonL, epsilona,
                 recovered_threshold, fraction_seed, mutant = True):
  tv = find_tv_var(y0, t, phi, theta, betaw, gammaw, betam, gammam, epsilonL, recovered_threshold)
  print(f"tv = {tv}")

  #if variant not found/we are not seeding
  if tv == None or mutant == False:
    sol_full = solve_ivp(instant_vacc_SIR, [0,t], y0,
                  args=(phi, theta, betaw, gammaw, betam, gammam, epsilonL),
                  t_eval = np.linspace(0, t, t))

    y_full = sol_full.y
    t_full = sol_full.t

  #if variant is found or we are seeding
  else:
    sol1 = solve_ivp(instant_vacc_SIR, [0,tv], y0,
                    args=(phi, theta, betaw, gammaw, betam, gammam, epsilonL),
                    t_eval = np.linspace(0, tv, t))

    #last timepoint at tv
    y_tv = sol1.y[:, -1]
    print(f"y_tv S = {y_tv[0]}, Iw = {y_tv[1]}, Im = {y_tv[5]}")
    print(f"y_tv Rw = {y_tv[2]}")

    #seed small fraction from Iw into Im
    y_tv[1] -= fraction_seed
    y_tv[5] += fraction_seed


    #second part of solution, after tv
    sol2 = solve_ivp(
        instant_vacc_SIR,
        [tv, t], # start at tv
        y_tv, # use updated initial condition
        args=(phi, theta, betaw, gammaw, betam, gammam, epsilonL),
        t_eval = np.linspace(tv, t, t)
    )

    #make into one full soln
    t_full = np.concatenate([sol1.t, sol2.t])
    y_full = np.hstack([sol1.y, sol2.y])

  return([t_full, y_full])

#MULTIPLE SOLN SOLVER, FOR SEEDED

def solve_aonleakyinter_vacc(VE, theta, phi, beta, gamma,
                             S0, Iw0, Im0, Rw0, Rm0, CIw0, CIm0,
                             recovered_threshold, fraction_seed,
                             t, mutant = True, inter = True):
  """
  Requires no vaccination from the beginning, in other words,
  S0 should not have N0 or P0 subtracted already
  """

  if mutant == False:
    Im_new = 0
    Iw_new = Iw0 + Im0
  else:
    Im_new = Im0
    Iw_new = Iw0

  N0 = 0
  P0 = 0

  #checker
  if Iw_new+S0+Rw0+N0+P0+Rm0+Im_new != 1:
    print(f"ERROR_ERROR: {Iw_new+S0+Rw0+N0+N0+Rm0+Im_new}")
    print(f"Iw_new:{Iw_new}; Im_new:{Im_new} S0:{S0}; Rw0:{Rw0}; Rm0:{Rm0}; N0:{N0}; N0:{N0};")

  y0 = [S0, Iw_new, Rw0, N0, P0, Im_new, Rm0, CIw0, CIm0]
  #y0 is the same for all mdoes, because it only changes when ppl are vaccinated
  #which occurs at seeding in solve_seeded_vacc

  # aon
  epsilona = VE
  epsilonL_aon = 1

  sol_aon = solve_seeded_vacc(y0, t, phi, theta, beta, gamma,
                              beta, gamma, epsilonL_aon, epsilona,
                              recovered_threshold, fraction_seed)

  # leaky
  epsilona = 1
  epsilonL_leaky = VE


  sol_leaky = solve_seeded_vacc(y0, t, phi, theta, beta, gamma,
                                beta, gamma, epsilonL_leaky, epsilona,
                                recovered_threshold, fraction_seed)

  # inter
  if inter == True:
    epsilona = math.sqrt(VE)
    epsilonL_inter = math.sqrt(VE)

    sol_inter = solve_seeded_vacc(y0, t, phi, theta, beta, gamma,
                                  beta, gamma, epsilonL_inter, epsilona,
                                  recovered_threshold, fraction_seed)

    return sol_aon, sol_leaky, sol_inter

  else:
    return sol_aon, sol_leaky,


def solve_aonleakyinter_var(VE, theta, phi, beta, gamma,
                            S0, Iw0, Im0, Rw0, Rm0, CIw0, CIm0,
                            recovered_threshold, fraction_seed,
                            t, mutant = True, inter = True):
  #make sure im0 starts at 0 no matter what
    Im_new = 0
    Iw_new = Iw0 + Im0

    CIw0 = Iw_new
    CIm0 = Im_new

    # aon
    epsilona = VE
    epsilonL_aon = 1

    N0 = (1-epsilona)*(1-(S0+Iw_new+Rw0+Rm0+Im_new))
    P0 = 1-(S0+Iw_new+Rw0+Rm0+Im_new+N0)

    #check to make sure it all adds up
    if Iw_new+S0+Rw0+N0+P0+Rm0+Im_new != 1:
      print(f"ERROR_ERROR_AON: {Iw_new+S0+Rw0+N0+N0+Rm0+Im_new}")
      print(f"Iw_new:{Iw_new}; Im_new:{Im_new} S0:{S0}; Rw0:{Rw0}; Rm0:{Rm0}; N0:{N0}; N0:{N0};")

    y0_aon = [S0, Iw_new, Rw0, N0, P0, Im_new, Rm0, CIw0, CIm0]

    sol_aon = solve_seeded_var(y0_aon, t, phi, theta, beta, gamma,
                               beta, gamma, epsilonL_aon, epsilona,
                               recovered_threshold, fraction_seed, mutant=mutant)

    # leaky
    epsilona = 1
    epsilonL_leaky = VE

    N0 = (1-epsilona)*(1-(S0+Iw_new+Rw0+Rm0+Im_new))
    P0 = 1-(S0+Iw_new+Rw0+Rm0+Im_new+N0)

    y0_leaky = [S0, Iw_new, Rw0, N0, P0, Im_new, Rm0, CIw0, CIm0]

    sol_leaky = solve_seeded_var(y0_leaky, t, phi, theta, beta, gamma,
                                 beta, gamma, epsilonL_leaky, epsilona,
                                 recovered_threshold, fraction_seed,mutant=mutant)

    if Iw_new+S0+Rw0+N0+P0+Rm0+Im_new != 1:
      print(f"ERRORERROR_LEAKY: {Iw_new+S0+Rw0+N0+P0+Rm0+Im_new}")
      print(f"Iw_new:{Iw_new}; Im_new:{Im_new} S0:{S0}; Rw0:{Rw0}; Rm0:{Rm0}; N0:{N0}; N0:{N0};")

    # inter
    if inter == True:
      epsilona = math.sqrt(VE)
      epsilonL_inter = math.sqrt(VE)

      N0 = (1-epsilona)*(1-(S0+Iw_new+Rw0+Rm0+Im_new))
      P0 = 1-(S0+Iw_new+Rw0+Rm0+Im_new+N0)

      y0_inter = [S0, Iw_new, Rw0, N0, P0, Im_new, Rm0, CIw0, CIm0]

      if Iw_new+S0+Rw0+N0+P0+Rm0+Im_new != 1:
        print(f"ERRORERROR_INTER: {Iw_new+S0+Rw0+N0+P0+Rm0+Im_new}")
        print(f"Iw_new:{Iw_new}; Im_new:{Im_new} S0:{S0}; Rw0:{Rw0}; Rm0:{Rm0}; N0:{N0}; N0:{N0};")

      sol_inter = solve_seeded_var(y0_inter, t, phi, theta, beta, gamma,
                                  beta, gamma, epsilonL_inter, epsilona,
                                  recovered_threshold, fraction_seed, mutant=mutant)
      return sol_aon, sol_leaky, sol_inter

    #not always get intermediate
    else:
      return sol_aon, sol_leaky

def decide_aonleaky_solver(VE, theta, phi, beta, gamma,
                           S0, Iw0, Im0, Rw0, Rm0, CIw0, CIm0, t,
                           vacc_or_var, seeded, recovered_threshold = None,
                           fraction_seed = None, mutant = True, ):
  '''
  Function to call correct solver (seeded or unseeded).
  '''

  if seeded: #use seeded solvers if seeded
    assert vacc_or_var in ('vacc', 'var'), "vacc_or_var must be 'vacc' or 'var'"
    if vacc_or_var == "var":
      sol_aon, sol_leaky = solve_aonleakyinter_var(
        VE, theta, phi, beta, gamma,
        S0, Iw0, Im0, Rw0, Rm0, CIw0, CIm0,
        recovered_threshold, fraction_seed,
        t, inter = False, mutant = mutant)

    else:
      sol_aon, sol_leaky = solve_aonleakyinter_vacc(
        VE, theta, phi, beta, gamma,
        S0, Iw0, Im0, Rw0, Rm0, CIw0, CIm0,
        recovered_threshold, fraction_seed,
        t, inter = False, mutant = mutant)

  else: #not seeded solver
    sol_aon, sol_leaky = solve_aonleaky(
      VE, theta, phi, beta, gamma,
      S0, Iw0, Im0, Rw0, Rm0, CIw0, CIm0, t, mutant = mutant)

  return sol_aon, sol_leaky

#combined heat map + max finder
def compute_measure_df(measure, Rnaughts=np.linspace(1.0, 3.0, 17), VEs=np.linspace(0.0, 1.0, 21),
                       VCs=[0.6, 1.0], gamma=1/4, t=300, theta=0.6, phi=0.6,
                       Iw0=0.0099, Im0=0.0001, Rw0=0, Rm0=0,
                       seeded=False, vacc_or_var=None,
                       recovered_threshold=0.15, fraction_seed=0.0001):
  """
  Core loops. Returns {VC: dataframe} for the given measure.
  """
  assert measure in ('delta_CI', 'delta_mutimpact', 'delta_VI'), "measure must be 'delta_CI, _mutimpact, or _VI'"

  if seeded:
    assert vacc_or_var in ('vacc', 'var'), "vacc_or_var must be 'vacc' or 'var'"

  results = {}

  for VC in VCs:
    rows = []

    #diff s0 conditions for var vs vacc
    if vacc_or_var == "var":
      S0 = (1-Iw0-Im0)*(1-VC)

    elif vacc_or_var == "vacc":
      S0 = 1-Iw0-Im0
      fraction_seed = VC

    else:
      S0 = (1-Iw0-Im0)*(1-VC)
      print(f"S0={S0}")

    CIw0 = Iw0
    CIm0 = Im0

    for Rnaught in Rnaughts:
      for VE in VEs:
        beta = Rnaught*gamma
        # SOLVING, BASIC W MUTANT CONDITIONS
        sol_aon, sol_leaky = decide_aonleaky_solver(VE, theta, phi, beta, gamma,
                           S0, Iw0, Im0, Rw0, Rm0, CIw0, CIm0, t,
                           vacc_or_var, seeded = seeded, recovered_threshold = recovered_threshold,
                           fraction_seed = fraction_seed, mutant = True)

        var = True if vacc_or_var == "var" else False

        #FOR DIFF IN MUTANT IMPACT MEASURE (needs no mutant solver)
        if measure == 'delta_mutimpact':
          nmsol_aon, nmsol_leaky = decide_aonleaky_solver(VE, theta, phi, beta, gamma,
                           S0, Iw0, Im0, Rw0, Rm0, CIw0, CIm0, t,
                           vacc_or_var, seeded = seeded, recovered_threshold = recovered_threshold,
                           fraction_seed = fraction_seed, mutant = False,)

          #actually solve for metric
          metric = compare_mutantimpact(sol_aon[1], sol_leaky[1], nmsol_aon[1],
                                        nmsol_leaky[1], recovered_threshold, var)

        #FOR DIFF IN CI MEASURE
        elif measure == 'delta_CI':
          metric = find_deltaCI(sol_aon[1], sol_leaky[1], recovered_threshold, var)

        elif measure == 'delta_VI':

          sol_nv = solve_nv(theta, phi, beta, gamma, Iw0, Im0, Rw0, Rm0, t,
                            nm_cf = False, var = var, recovered_threshold = recovered_threshold,
                            fraction_seed = fraction_seed)

          metric = find_deltaVI(sol_aon[1], sol_leaky[1], sol_nv[1], recovered_threshold, var = var)

        rows.append((round(VE, 2), Rnaught, metric))



    #append row to VC df for VE and R0
    results[VC] = pd.DataFrame(rows, columns=["VE", "Rnaught", measure])

  return results


def get_max_deltameasure(measure, return_df=False, **kwargs):
    """
    Returns {VC: {"max", "VE", "Rnaught"}}.
    Set return_df=True to also get the full dataframe back e.g. for plotting.
    """
    dfs = compute_measure_df(measure, **kwargs)
    results = {}

    for VC, df in dfs.items():
        max_row = df.loc[df[measure].idxmax()]
        results[VC] = {"max":     max_row[measure],
                       "VE":      max_row["VE"],
                       "Rnaught": max_row["Rnaught"]}

    if return_df:
        return results, dfs
    return results


def plot_heatmap(measure, Rnaughts=np.linspace(1.0, 3.0, 17), VEs=np.linspace(0.0, 1.0, 21),
                 VCs=[0.6, 1.0], gamma=1/4, t = int(200), theta=0.6, phi=0.6,
                 Iw0=0.0099, Im0=0.0001, Rw0=0, Rm0=0,
                 cmap=None, vmin=None, vmax=None, norm=None,
                 seeded=False, vacc_or_var=None,
                 recovered_threshold=0.15, fraction_seed=0.0001,
                 return_max=False, suptitle = None):
    """
    Plots heatmap. Set return_max=True to also return max values without looping twice.
    """
    dfs = compute_measure_df(measure, Rnaughts, VEs, VCs, gamma, t, theta, phi,
                             Iw0, Im0, Rw0, Rm0, seeded, vacc_or_var,
                             recovered_threshold, fraction_seed)

    cmap_defaults = {'delta_CI': dict(cmap="rocket_r", vmin=0,  vmax=0.25),
                     'delta_VI': dict(cmap="rocket_r", vmin=0,  vmax=25),
                     'delta_mutimpact': dict(cmap="PuOr",     vmin=-3, vmax=3)}

    titles = {'delta_CI': "Difference in Cumulative Infections",
              'delta_VI': "Difference in Vaccine Impact",
              'delta_mutimpact': "Difference in Prop. Increase in CI relative to no-mutant"}

    cmap_settings = cmap_defaults[measure].copy()

    if cmap is not None: cmap_settings['cmap'] = cmap
    if vmin is not None: cmap_settings['vmin'] = vmin
    if vmax is not None: cmap_settings['vmax'] = vmax
    if norm is not None:
        cmap_settings.pop('vmin', None)
        cmap_settings.pop('vmax', None)
        cmap_settings['norm'] = norm

    plt.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, VC in zip(axes, VCs):
        df = dfs[VC]
        heatmap = df.pivot(index="Rnaught", columns="VE", values=measure)

        sns.heatmap(heatmap, ax=ax, **cmap_settings)

        ax.set_title(f"Vaccine coverage = {VC}", pad=10, fontsize=24)
        ax.set_xlabel("Vaccine efficacy", fontsize=24)
        ax.set_ylabel("R0, basic reproduction number", fontsize=24)
        ax.invert_yaxis()

        xtick_positions = list(range(0, len(VEs), 4))
        xtick_labels = ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']
        ax.set_xticks([p + 0.5 for p in xtick_positions])
        ax.set_xticklabels(xtick_labels, rotation=0, fontsize=15)

        ytick_positions = [i for i, r in enumerate(Rnaughts) if round(r * 2) == r * 2]
        ytick_labels = [f"{Rnaughts[i]:.1f}" for i in ytick_positions]
        ax.set_yticks([p + 0.5 for p in ytick_positions])
        ax.set_yticklabels(ytick_labels, rotation=0, fontsize=15)

        ax.tick_params(axis='y', rotation=0, labelsize=15)
        ax.tick_params(axis='x', rotation=0, labelsize=15)

        #set gray nan
        ax.collections[0].cmap.set_bad('0.7')


    if vacc_or_var != "var":
      fraction_seed = None #so it does not appear in suptitle if vaccination (its VC) or unseeded
      if seeded == False:
        recovered_threshold = None

    suptitle_created = create_suptitle(gamma, theta=theta, Im0=Im0, Iw0=Iw0,
                               recovered_threshold = recovered_threshold,
                               fraction_seed = fraction_seed, suptitle = suptitle)
    fig.suptitle(f"{titles[measure]}\n{suptitle_created}", fontsize=20, y=1.02)
    plt.tight_layout()
    plt.show()

    if return_max:
        max_results = {}
        for VC, df in dfs.items():
            max_row = df.loc[df[measure].idxmax()]
            max_results[VC] = {"max": max_row[measure],
                               "VE": max_row["VE"],
                               "Rnaught": max_row["Rnaught"]}
        return max_results

#HEATMAPS FOR WT OR MUT ONLY
def find_deltaCIw(solution_aon, solution_leaky, recovered_threshold = None, var = False):
  """
  Calculate difference in total cumulative infections between aon and leaky solutions
  """
  if var == True:
    if check_threshold_modes(solution_aon, solution_leaky, recovered_threshold) == True:
      return np.nan

  CI_aon = find_finalCIw(solution_aon) 
  CI_leaky = find_finalCIw(solution_leaky) 

  return CI_leaky - CI_aon

def find_deltaCIm(solution_aon, solution_leaky, recovered_threshold = None, var = False):
  """
  Calculate difference in total cumulative infections between aon and leaky solutions
  """
  if var == True:
    if check_threshold_modes(solution_aon, solution_leaky, recovered_threshold) == True:
      return np.nan

  CI_aon = find_finalCIm(solution_aon) 
  CI_leaky = find_finalCIm(solution_leaky) 

  return CI_leaky - CI_aon


#combined heat map + max finder
def compute_CIs_df(measure, Rnaughts=np.linspace(1.0, 3.0, 17), VEs=np.linspace(0.0, 1.0, 21),
                       VCs=[0.6, 1.0], gamma=1/4, t=300, theta=0.6, phi=0.6,
                       Iw0=0.0099, Im0=0.0001, Rw0=0, Rm0=0,
                       seeded=False, vacc_or_var=None,
                       recovered_threshold=0.15, fraction_seed=0.0001):
  """
  Core loops. Returns {VC: dataframe} for the given measure.
  """
  assert measure in ('delta_CIw', 'delta_CIm',), "measure must be 'delta_CIw', 'delta_CIm'"

  if seeded:
    assert vacc_or_var in ('vacc', 'var'), "vacc_or_var must be 'vacc' or 'var'"

  results = {}

  for VC in VCs:
    rows = []

    #diff s0 conditions for var vs vacc
    if vacc_or_var == "var":
      S0 = (1-Iw0-Im0)*(1-VC)

    elif vacc_or_var == "vacc":
      S0 = 1-Iw0-Im0
      fraction_seed = VC

    else:
      S0 = (1-Iw0-Im0)*(1-VC)
      print(f"S0={S0}")

    CIw0 = Iw0
    CIm0 = Im0

    for Rnaught in Rnaughts:
      for VE in VEs:
        beta = Rnaught*gamma
        # SOLVING, BASIC W MUTANT CONDITIONS
        sol_aon, sol_leaky = decide_aonleaky_solver(VE, theta, phi, beta, gamma,
                           S0, Iw0, Im0, Rw0, Rm0, CIw0, CIm0, t,
                           vacc_or_var, seeded = seeded, recovered_threshold = recovered_threshold,
                           fraction_seed = fraction_seed, mutant = True)

        var = True if vacc_or_var == "var" else False

        #FOR DIFF IN CI MEASURE
        if measure == 'delta_CIw':
          metric = find_deltaCIw(sol_aon[1], sol_leaky[1], recovered_threshold, var)
        elif measure == 'delta_CIm':
          metric = find_deltaCIm(sol_aon[1], sol_leaky[1], recovered_threshold, var)
        
        rows.append((round(VE, 2), Rnaught, metric))

    #append row to VC df for VE and R0
    results[VC] = pd.DataFrame(rows, columns=["VE", "Rnaught", measure])

  return results


def get_max_deltameasure(measure, return_df=False, **kwargs):
    """
    Returns {VC: {"max", "VE", "Rnaught"}}.
    Set return_df=True to also get the full dataframe back e.g. for plotting.
    """
    dfs = compute_measure_df(measure, **kwargs)
    results = {}

    for VC, df in dfs.items():
        max_row = df.loc[df[measure].idxmax()]
        results[VC] = {"max":     max_row[measure],
                       "VE":      max_row["VE"],
                       "Rnaught": max_row["Rnaught"]}

    if return_df:
        return results, dfs
    return results


def plot_heatmap_CI(measure, Rnaughts=np.linspace(1.0, 3.0, 17), VEs=np.linspace(0.0, 1.0, 21),
                 VCs=[0.6, 1.0], gamma=1/4, t = int(200), theta=0.6, phi=0.6,
                 Iw0=0.0099, Im0=0.0001, Rw0=0, Rm0=0,
                 cmap=None, vmin=None, vmax=None, norm=None,
                 seeded=False, vacc_or_var=None,
                 recovered_threshold=0.15, fraction_seed=0.0001,
                 return_max=False, suptitle = None):
    """
    Plots heatmap. Set return_max=True to also return max values without looping twice.
    """
    dfs = compute_CIs_df(measure, Rnaughts, VEs, VCs, gamma, t, theta, phi,
                             Iw0, Im0, Rw0, Rm0, seeded, vacc_or_var,
                             recovered_threshold, fraction_seed)

    cmap_defaults = {'delta_CIw': dict(cmap="rocket_r", vmin=0,  vmax=0.15),
                     'delta_CIm': dict(cmap="rocket_r", vmin=0,  vmax=0.15),}

    titles = {'delta_CIw': "Difference in Cumulative Infections",
              'delta_CIm': "Difference in Vaccine Impact"}
    
    cmap_settings = cmap_defaults[measure].copy()

    if cmap is not None: cmap_settings['cmap'] = cmap
    if vmin is not None: cmap_settings['vmin'] = vmin
    if vmax is not None: cmap_settings['vmax'] = vmax
    if norm is not None:
        cmap_settings.pop('vmin', None)
        cmap_settings.pop('vmax', None)
        cmap_settings['norm'] = norm

    plt.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, VC in zip(axes, VCs):
        df = dfs[VC]
        heatmap = df.pivot(index="Rnaught", columns="VE", values=measure)

        sns.heatmap(heatmap, ax=ax, **cmap_settings)

        ax.set_title(f"Vaccine coverage = {VC}", pad=10, fontsize=24)
        ax.set_xlabel("Vaccine efficacy", fontsize=24)
        ax.set_ylabel("R0, basic reproduction number", fontsize=24)
        ax.invert_yaxis()

        xtick_positions = list(range(0, len(VEs), 4))
        xtick_labels = ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']
        ax.set_xticks([p + 0.5 for p in xtick_positions])
        ax.set_xticklabels(xtick_labels, rotation=0, fontsize=15)

        ytick_positions = [i for i, r in enumerate(Rnaughts) if round(r * 2) == r * 2]
        ytick_labels = [f"{Rnaughts[i]:.1f}" for i in ytick_positions]
        ax.set_yticks([p + 0.5 for p in ytick_positions])
        ax.set_yticklabels(ytick_labels, rotation=0, fontsize=15)

        ax.tick_params(axis='y', rotation=0, labelsize=15)
        ax.tick_params(axis='x', rotation=0, labelsize=15)

        #set gray nan
        ax.collections[0].cmap.set_bad('0.7')


    if vacc_or_var != "var":
      fraction_seed = None #so it does not appear in suptitle if vaccination (its VC) or unseeded
      if seeded == False:
        recovered_threshold = None

    suptitle_created = create_suptitle(gamma, theta=theta, Im0=Im0, Iw0=Iw0,
                               recovered_threshold = recovered_threshold,
                               fraction_seed = fraction_seed, suptitle = suptitle)
    fig.suptitle(f"{titles[measure]}\n{suptitle_created}", fontsize=20, y=1.02)
    plt.tight_layout()
    plt.show()

    if return_max:
        max_results = {}
        for VC, df in dfs.items():
            max_row = df.loc[df[measure].idxmax()]
            max_results[VC] = {"max": max_row[measure],
                               "VE": max_row["VE"],
                               "Rnaught": max_row["Rnaught"]}
        return max_results
