import seaborn as sns
from itertools import chain
from itertools import product
from scipy.stats import chi2
from scipy.stats import chi2_contingency
from CorrespondenceAnalysisPy.correspondence_analysis_computation import ca_compute
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scipy.stats as ss
import scipy.stats as stats
import researchpy as rp
import time
import qgrid
from pivottablejs import pivot_ui
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import re
import pandas_profiling
from ipywidgets import fixed, interactive, interact_manual, interact, IntProgress
import ipywidgets as widgets
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
sns.set()
#from plotly import tools


# need to activate
# jupyter nbextension enable --py --sys-prefix widgetsnbextension
# jupyter nbextension enable --py --sys-prefix qgrid

#import cufflinks as cf


init_notebook_mode(connected=True)
alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
            'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'AA', 'AB',
            'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI']
# Define extra functions


def create_all_keys(df, date, method=1):
    """

    """
    l_cont = list(df.select_dtypes(include=['int', 'float']))
    l_cat = list(df.select_dtypes(include='object'))

    l_low = []
    l_high = []
    for x in l_cat:
        count = df[x].nunique()
        if count > 10:
            l_high.append(x)
        else:
            l_low.append(x)

    if method == 1:
        dic_ts = {
            'var_date': {
                'V0': {
                    'name': date,
                    'variable_db': date,
                    'Drop': []
                }
            },
            'var_continuous': {}
        }
        for i, var_cont in enumerate(l_cont):

            key = 'V' + str(i)
            dic_ts['var_continuous'][key] = {
                'name': var_cont,
                'variable_db': var_cont,
                # 'categorical_low': ['Post'],
                'Drop': [],
                'drop_decile': [],
                'drop_value': []
            }
        return dic_ts
    elif method == 2:
        dic_Low = {'var_continuous': {}, 'var_categorical_low': {}}

        for i, cont in enumerate(l_cont):
            key = 'V' + str(i)
            dic_Low['var_continuous'][key] = {
                'name': cont,
                'variable_db': cont,
                'drop_decile': [],
                'drop_value': []
            }

        for i, cat in enumerate(l_low):
            key = 'V' + str(i)
            dic_Low['var_categorical_low'][key] = {
                'name': cat,
                'variable_db': cat,
                'Drop': []
            }

        return dic_Low
    elif method == 3:
        dic_high = {'var_continuous': {}, 'var_categorical_high': {}}

        for i, cont in enumerate(l_cont):
            key = 'V' + str(i)
            dic_high['var_continuous'][key] = {
                'name': cont,
                'variable_db': cont,
                'drop_decile': [],
                'drop_value': []
            }

        for i, cat in enumerate(l_high):
            key = 'V' + str(i)
            dic_high['var_categorical_high'][key] = {
                'name': cat,
                'variable_db': cat,
                'Drop': []
            }

        return dic_high
    elif method == 4:

        dic_highLow = {
            'var_continuous': {},
            'var_categorical_high': {},
            'var_categorical_low': {}
        }

        for i, cont in enumerate(l_cont):
            key = 'V' + str(i)
            dic_highLow['var_continuous'][key] = {
                'name': cont,
                'variable_db': cont,
                'drop_decile': [],
                'drop_value': []
            }

        for i, cat in enumerate(l_high):
            key = 'V' + str(i)
            dic_highLow['var_categorical_high'][key] = {
                'name': cat,
                'variable_db': cat,
                'Drop': []
            }

        for i, cat in enumerate(l_low):
            key = 'V' + str(i)
            dic_highLow['var_categorical_low'][key] = {
                'name': cat,
                'variable_db': cat,
                'Drop': []
            }
        return dic_highLow
    elif method == 5:
        dic_scatter = {'var_Y': {}, 'var_X': {}}

        for i, cont in enumerate(l_cont):
            key = 'V' + str(i)

            dic_scatter['var_Y'][key] = {
                'name': cont,
                'variable_db': cont,
                'drop_decile': [],
                'drop_value': []
            }

        for i, cont in enumerate(l_cont):
            key = 'V' + str(i)

            dic_scatter['var_X'][key] = {
                'name': cont,
                'variable_db': cont,
                'drop_decile': [],
                'drop_value': []
            }

        return dic_scatter
    elif method == 6:

        dic_scatterG = {'var_Y': {}, 'var_X': {}, 'var_grouping': {}}

        for i, cont in enumerate(l_cont):
            key = 'V' + str(i)

            dic_scatterG['var_Y'][key] = {
                'name': cont,
                'variable_db': cont,
                'drop_decile': [],
                'drop_value': []
            }

        for i, cont in enumerate(l_cont):
            key = 'V' + str(i)

            dic_scatterG['var_X'][key] = {
                'name': cont,
                'variable_db': cont,
                'drop_decile': [],
                'color': [],
                'drop_value': []
            }
        for i, cat in enumerate(l_cat):
            key = 'V' + str(i)

            dic_scatterG['var_grouping'][key] = {
                'name': cat,
                'variable_db': cat,
                'Drop': []
            }

        return dic_scatterG

    elif method == 7:
        dic_cat = {'var_columns': {}, 'var_rows': {}}

        for i, cat in enumerate(l_cat):
            key = 'V' + str(i)

            dic_cat['var_columns'][key] = {
                'name': cat,
                'variable_db': cat,
                'Drop': []
            }

        for i, cat in enumerate(l_cat):
            key = 'V' + str(i)

            dic_cat['var_rows'][key] = {
                'name': cat,
                'variable_db': cat,
                'values': [],
                'Drop': []
            }
        return dic_cat


def highlight_reject(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_true = s == True
    return ['background-color: yellow' if v else '' for v in is_true]


def random_color(s):
    """
    The function returns a random RGB color;
    It helps to get random colors since the number of variables is not known
    """
    import random
    number_of_colors = s
    color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j
                            in range(6)])
             for i in range(number_of_colors)]
    return color


def setcolors(list_group):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    list_color = []
    for g in list_group:
        color = random_color(1)[0]
        list_color.append([g, color])
    return list_color


def applycolors(val, l_colors):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    for x in l_colors:
        if val in x:
            color = x[1]
    return 'background-color: %s' % color


def regress(X_name, Y_name, group, x, y):

    #pearson = np.corrcoef(x, y)[0][1]
    n = len(x)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    dic_res = {
        'Y': Y_name,
        'X': X_name,
        'group': group,
        'n_rows': [n],
        'pearson/R2': [r_value],
        'slope': [slope],
        'intercept': [intercept],
        'p_value': [p_value],
        'std_err': [std_err],
    }

    df = pd.DataFrame(dic_res)
    return df


def hexbin(x, y, color, **kwargs):
    """
    """
    cmap = sns.light_palette(color, as_cmap=True)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['font.serif'] = ['SimHei']
    plt.hexbin(x, y, gridsize=10, cmap=cmap, **kwargs)

# Initialize Google drive cdr

#############################################################################
#############################################################################
#############################################################################
##################### QUICKSTART ##############


def quickstat(df,
              export=False,
              move_to_drive=False,
              name="output",
              folder=False,
              cdr=False):
    """
    The function compute the summary statistic of the dataframe from
    pandas's pandas_profiling function.

    The use can export the report in html format either locally or in Google
    drive
    """
    profile = pandas_profiling.ProfileReport(df)

    if export:
        name_html = name + '.html'
        profile.to_file(name_html)
        path = os.getcwd()
        print('File {0} saved at this path {1}'.format(name_html, path))
        if move_to_drive:
            mime_type = "text/html"
            cdr.upload_file_root(mime_type, name_html)
            cdr.move_file(file_name=name_html,
                          folder_name=folder)
            os.remove(name_html)

    return display(profile)


def make_quickstart(df, cdr=False):
    """
    This function use IPython widget interactive to allow interactivity
    with the function quickstat.
    The user can choose to:
    - export the ProfileReport
    - Move the report to Google Drive
    - Choose a name for the report
    - Choose the folder in Google drive to save the report
    """
    return interactive(quickstat,
                       {"manual": True, "auto_display": False},
                       df=fixed(df),
                       name='',
                       folder='',
                       export=False,
                       move_to_drive=False,
                       cdr=fixed(cdr)
                       )


#############################################################################
#############################################################################
#############################################################################
##################### Grid search ##############

def grid_search(df, rows_to_display=10):
    """
    """

    return qgrid.show_grid(df,
                           grid_options={
                               # SlickGrid options
                               'fullWidthRows': False,
                               'syncColumnCellResize': False,
                               'forceFitColumns': False,
                               'defaultColumnWidth': 150,
                               'rowHeight': 28,
                               'enableColumnReorder': False,
                               'enableTextSelectionOnCells': True,
                               'editable': True,
                               'autoEdit': False,
                               'explicitInitialization': True,

                               # Qgrid options
                               'maxVisibleRows': rows_to_display,
                               'minVisibleRows': 5,
                               'sortable': True,
                               'filterable': True,
                               'highlightSelectedCell': False,
                               'highlightSelectedRow': True
                           }
                           )

#############################################################################
#############################################################################
#############################################################################
##################### Pivot table ##############


def pivot_table(df):
    """

    """
    return pivot_ui(df)

#############################################################################
#############################################################################
#############################################################################
##################### TIME SERIE ##############


def list_dropdownTS(dic_df):
    """
    input a dictionary containing what variables to use, and how to clean
    the variables
    It outputs a list with the possible pair solutions.
    This function will populate a dropdown menu in the eventHandler function
    """

    l_choice = []
    for key_cat, value_cat in dic_df['var_continuous'].items():
        l_choice.append(value_cat['name'])
    l_choice = ['-'] + l_choice

    return l_choice


def computation_ts(df, dic_df, index_var, group=False, log=False):
    """
    Core function for the tab Time Series.
    It inputs:
    - Dataframe
    - dictionnary with action and variables to perform
    - The index of the dictionnary. It can be V0, V1, and so on

    It ouputs:
    - y:dataframe with the mean, median and sum of the continuous variables
    - df_var: containes the raw data
    - var_date: name of the date variable
    - var_continuous: name of the continuous in the dataframe
    - name_continuous: new name of the continuous variable
    - name_date: name of the date variable
    - df_melt: melted y dataframe: use to plot the graph with seaborn. We want the mean and
    median on the same graph


    """

    # 1: Extract the variable name from the dictionnary
    index_date = 'V0'

    var_continuous = dic_df['var_continuous'][index_var]['variable_db']
    name_continuous = dic_df['var_continuous'][index_var]['name']

    var_date = dic_df['var_date'][index_date]['variable_db']
    name_date = dic_df['var_date'][index_date]['name']

    drop_date = dic_df['var_date'][index_date]['Drop']
    drop_categorical = dic_df['var_continuous'][index_var]['Drop']
    drop_continuous = dic_df['var_continuous'][
        index_var]['drop_value']
    drop_decile = dic_df['var_continuous'][
        index_var]['drop_decile']

    # 2 Select the variables

    if group != False:
        df_var = df[[var_date, var_continuous, group]]
    else:
        df_var = df[[var_date, var_continuous]]

    # 3 Clean the dataframe:
    # drop continuous: possibility to drop unexpected value like 0, NAN etc

    if len(drop_continuous) > 0:
        # print(drop_continuous)
        df_var = df_var.loc[~df_var.isin(drop_continuous).any(1)]

    # drop outliers
    if len(drop_decile) > 0:
                            # print(drop_decile)
        lower_d = drop_decile[0]
        value_lower_d = df_var[var_continuous].quantile(lower_d)

        higher_d = drop_decile[1]
        value_higher_d = df_var[var_continuous].quantile(higher_d)

        df_var = df_var[df_var[var_continuous] > value_lower_d]
        df_var = df_var[df_var[var_continuous] < value_higher_d]

        df_decile = pd.DataFrame({'Decile': [lower_d, higher_d],
                                  'Value': [value_lower_d, value_higher_d]})

    # drop categorical value in the date
    if len(drop_date) > 0:
        df_var = df_var.loc[~df_var.isin(drop_date).any(1)]

    # drop categorical value in the categorical columns
    if len(drop_categorical) > 0:
        # print(drop_categorical)
        df_var = df_var.loc[~df_var.isin(drop_categorical).any(1)]

    if log:
        df_1 = df_var.copy()
        df_1[var_continuous] = np.log(df_1[var_continuous])
    else:
        df_1 = df_var.copy()

    # 4 Compute mean/median/sum by date
    y = df_1.groupby(var_date).agg({
        var_continuous: ['mean',
                         'median',
                         'sum']})
    y.columns = y.columns.droplevel()
    y = y.reset_index()

    # 5 Compute the difference

    y['diff'] = y['sum'].diff()

    df_melt = pd.melt(y, id_vars=[var_date], value_vars=['median',
                                                         'mean'],
                      var_name='stat',
                      value_name=name_continuous)

    dic_int = {
        'output': [y, df_1, var_date, var_continuous, name_continuous,
                   name_date, df_melt]
    }

    return dic_int


def saveToDriveTS(cdr, sheetID, sheetName, folder, y, LatestRow,
                  df_var, var_date,
                  name_continuous, group, var_continuous, df_melt):
    """
    """
    fig, axarr = plt.subplots(1, 2, figsize=(12, 8))
    g = sns.lineplot(x=var_date,
                     y=name_continuous,
                     hue=var_continuous,
                     data=df_melt, ax=axarr[0])
    g = sns.lineplot(x=var_date,
                     y='sum',
                     data=y, ax=axarr[1])
    fig.suptitle('Mean/median & Sum of ' + name_continuous)

    y = y.fillna('')
    n_rows = y.shape[0]
    nb_cols = y.shape[1] + 2
    nb_rows = LatestRow + n_rows + 6
    # Rename columns
    y.columns = [var_date, 'mean_' + name_continuous,
                 'median_' + name_continuous,
                 'sum_' + name_continuous,
                 'diff'
                 ]

    for i, letter in enumerate(alphabet):
        if i == nb_cols:
            range_2_letter = letter
        if i + 2 == nb_cols:
            range_1_letter = letter

    if group != False:
        y_g = df_var.groupby([var_date, group]).agg({
            var_continuous: ['mean',
                             'median',
                             'sum']})
        y_g.columns = y_g.columns.droplevel()
        y_g = y_g.reset_index()

        y_g.columns = [var_date,
                       group,
                       'mean_' + name_continuous,
                       'median_' + name_continuous,
                       'sum_' + name_continuous
                       ]
        dic_range = {

            'range_ts': "A" + str(LatestRow + 4) + ':' + str(range_1_letter) +
            str(nb_rows),
            'range_ts_g': "G" + str(LatestRow + 4) + ':Z' +
            str(LatestRow + y_g.shape[0] + 6)
        }
        # make graph with seaborn as save image to drive
        dic_df = {
            'df_ts': y.to_numpy().tolist(),
            'df_tsg': y_g.to_numpy().tolist(),
            'headers_ts': list(y),
            'header_tsg': list(y_g),
            'range_ts': dic_range
        }
        cdr.add_data_to_spreadsheet(data=dic_df['df_tsg'],
                                    sheetID=sheetID,
                                    sheetName=sheetName,
                                    rangeData=dic_df['range_ts']['range_ts_g'],
                                    headers=dic_df['header_tsg'])
    else:
        dic_range = {

            'range_ts': "A" + str(LatestRow + 4) + ':' + str(range_1_letter) +
            str(nb_rows)
        }

        dic_df = {
            'df_ts': y.to_numpy().tolist(),
            'headers_ts': list(y),
            'range_ts': dic_range
        }
    cdr.add_data_to_spreadsheet(data=dic_df['df_ts'],
                                sheetID=sheetID,
                                sheetName=sheetName,
                                rangeData=dic_df['range_ts']['range_ts'],
                                headers=dic_df['headers_ts'])
    name_ = 'Mean-median_&_Sum_of_' + name_continuous + '.png'
    g.get_figure().savefig(name_)
    mime_type = "image/png"
    cdr.upload_file_root(mime_type, name_)
    cdr.move_file(file_name=name_,
                  folder_name=folder)
    os.remove(name_)


def time_series_gs(df,
                   dic_multiple,
                   variable=False,
                   group=False,
                   log=False,
                   sheetID=False,
                   sheetName=False,
                   folder=False,
                   move_to_drive=False,
                   move_to_drive_batch=False,
                   cdr=False,
                   verbose=True):
    """
    The function inputs:
    - a dataframe
    - a dictionnary with the variables and actions to perform
    - variable: Name of the variable to plot the time serie
    - group: a variable to compute mean/median/sum by group
    - sheetid: sheet ID of Google Spreadsheet
    - sheetName:  Sheet name to paste the date
    - folder: Folder name to save the pictures
    - move_to_drive: Boolean: copy to drive?
    - verbose: Plot the graphs inside Jupyter

    The function uses the output of the function computation_ts to create the graphs
    and or paste the data into Google Spreadsheet

    If verbose true, the function creates a tab widget with one or two tabs, if group is defined
    If Google drive is true, the function parses all the variables in the dictionary
    """

    if move_to_drive_batch:
        max_ = len(dic_multiple['var_continuous'])
        # instantiate the bar
        f = IntProgress(min=0, max=max_ + 1, description='Loading:')
        display(f)

        # Loop over all the variables in var_continuous
        for key, value in dic_multiple['var_continuous'].items():
            variable = value['name']

            # increment the progressbar
            f.value += 1

        # Get the index of the dictionary using the value of the name of the column chosen
            for key, value in dic_multiple['var_continuous'].items():
                if str(variable) == value['name']:
                    index_dic_g2 = key

        # Make the computation
            index_date = 'V0'

            # COmpute data
            temp_comp = computation_ts(df=df, dic_df=dic_multiple,
                                       index_var=index_dic_g2,
                                       group=group,
                                       log=log)

           # Extract the output from the computation
            y = temp_comp['output'][0]
            df_var = temp_comp['output'][1]
            var_date = temp_comp['output'][2]
            var_continuous = temp_comp['output'][3]
            name_continuous = temp_comp['output'][4]
            df_melt = temp_comp['output'][6]

        # Make seaborn graph: includes mean + median in a single graph
        # right side of the grah is the sum
        # Loop over each value of  var continuous
        # for key, value in dic_ts['var_continuous'].items():
        # Need to open the spreadsheet to know the latest none
        # empty cell
        # Objective of moving to drive is to do batch computation
            LatestRow = cdr.getLatestRow(sheetID=sheetID,
                                         sheetName=sheetName)

            saveToDriveTS(
                cdr=cdr,
                sheetID=sheetID,
                sheetName=sheetName,
                folder=folder,
                y=y,
                LatestRow=LatestRow,
                df_var=df_var,
                var_date=var_date,
                name_continuous=name_continuous,
                group=group,
                var_continuous=var_continuous,
                df_melt=df_melt)

        f.value += 1

    # Create the tab and plot it in the jupyter
    if verbose:
        for key, value in dic_multiple['var_continuous'].items():
            if variable == value['name']:
                index_dic_g2 = key

        temp_comp = computation_ts(df=df, dic_df=dic_multiple,
                                   index_var=index_dic_g2,
                                   group=group,
                                   log=log)

        y = temp_comp['output'][0]
        df_var = temp_comp['output'][1]
        var_date = temp_comp['output'][2]
        var_continuous = temp_comp['output'][3]
        name_continuous = temp_comp['output'][4]
        name_date = temp_comp['output'][5]
        df_melt = temp_comp['output'][6]

        if move_to_drive:
            # To complicated to save pictures with Plotly locally
            LatestRow = cdr.getLatestRow(sheetID=sheetID,
                                         sheetName=sheetName)
            saveToDriveTS(
                cdr=cdr,
                sheetID=sheetID,
                sheetName=sheetName,
                folder=folder,
                y=y,
                LatestRow=LatestRow,
                df_var=df_var,
                var_date=var_date,
                name_continuous=name_continuous,
                group=group,
                var_continuous=var_continuous,
                df_melt=df_melt)

        # Define theme of plotly
        # cf.go_offline()
        #cf.set_config_file(offline=False, world_readable=False, theme='space')

        summary_plot = widgets.Output()

        if group != False:
            # Bug with plotly, cannot have two plotly graphs in two differents tabs
            # therefore, we use seaborn

            summary_plot_group = widgets.Output()

            df_group = df_var.groupby([var_date, group]).sum().reset_index()

            # df_group = df_group.pivot(index=var_date,
            #						  columns=group, values=var_continuous)

            tab_contents = [summary_plot_group, summary_plot]
            tab = widgets.Tab(tab_contents)
            tab.set_title(0, 'Plot time series, group')
            tab.set_title(1, 'Plot time series')
        else:
            #accordion = widgets.Accordion(children=[summary_plot, summary_plot_mean_med])
            #accordion.set_title(0, 'Sum')
            #accordion.set_title(1, 'Mean/median')

            tab_contents = [summary_plot]
            tab = widgets.Tab(tab_contents)
            tab.set_title(0, 'Plot time series')
        if move_to_drive != True:
            display(tab)

        if group != False:
            fig, axarr = plt.subplots(1, 2, figsize=(12, 8))
            g = sns.lineplot(x=var_date,
                             y=name_continuous,
                             hue='stat',
                             data=df_melt, ax=axarr[0])
            g = sns.lineplot(x=var_date,
                             y='sum',
                             data=y, ax=axarr[1])
            fig.suptitle('Mean/median & Sum of ' + name_continuous)

            with summary_plot:
                plt.show()
            # Need to fix the issue. Plotly cannot go inside tab

            with summary_plot_group:
                # df_group.iplot(
                #	asFigure=True,
                #	kind='scatter',
                #	xTitle=var_date,
                #	yTitle=var_continuous,
                #	title='Evolution of ' + name_continuous + ' by ' + group)
                # print(df_group)
                fig = px.line(df_group,
                              x=var_date,
                              y=name_continuous,
                              color=group,
                              template="plotly_dark")
                fig.update_traces(textposition='top center')
                fig.update_layout(
                    height=800,
                    title_text='Evolution of ' + name_continuous + ' by ' + group)

                fig.show()

        else:
            with summary_plot:
                # plt.show(fig)
                #y = y.set_index(var_date)
                #df_plot_up = y[['sum']]
                #df_plot_middle = y[['diff']]
                #df_plot_down = y[['mean', 'median']]

                fig_1 = px.line(y,
                                x=var_date,
                                y='sum',
                                template="plotly_dark")
                fig_1.update_traces(textposition='top center')
                fig_1.update_layout(
                    height=800,
                    title_text='Evolution of sum ' + name_continuous)

                fig_2 = px.line(y,
                                x=var_date,
                                y='mean',
                                template="plotly_dark")
                fig_2.update_traces(textposition='top center')
                fig_2.update_layout(
                    height=800,
                    title_text='Evolution of mean ' + name_continuous)

                fig_3 = px.line(y,
                                x=var_date,
                                y='median',
                                template="plotly_dark")
                fig_3.update_traces(textposition='top center')
                fig_3.update_layout(
                    height=800,
                    title_text='Evolution of median ' + name_continuous)

                fig_4 = px.bar(y,
                               x=var_date,
                               y='diff',
                               template="plotly_dark")
                fig_4.update_traces(textposition='auto')
                fig_4.update_layout(
                    height=800,
                    title_text=name_date + ' Difference of ' + name_continuous)
                #fig.update_traces(textposition='top center')
                # fig.update_layout(
                #	height=800,
                #	title_text= 'Evolution of ' + name_continuous)

                fig_1.show()
                fig_2.show()
                fig_3.show()
                fig_4.show()

                # df_plot_up.iplot(subplots=False,
                #				 yTitle=name_continuous,
                #				 title='Evolution of ' + name_continuous)

                # df_plot_up.iplot(subplots=False,
                #				 kind='bar',
                #				 yTitle=name_continuous,
                #				 title=name_date + ' Difference of ' + name_continuous)
            # with summary_plot_mean_med:

                # df_plot_down.iplot(subplots=True,
                #				   yTitle=name_continuous,
                #				   title='Evolution of ' + name_continuous,
                #				   shape=(2, 1))


def select_TS_eventHandler(df, dic_df, cdr=False):
    """
    Run the interactive input for the time serie function
    """

    l_filter = list(df.select_dtypes(include='object'))
    l_filter = [False] + l_filter

    x_widget = widgets.Dropdown(
        options=list_dropdownTS(dic_df=dic_df),
        value='-',
        description='Variable:')

    return interactive(time_series_gs,
                       {"manual": True, "auto_display": False},
                       df=fixed(df),
                       dic_multiple=fixed(dic_df),
                       variable=x_widget,
                       group=l_filter,
                       log=False,
                       sheetID='',
                       sheetName='',
                       folder='',
                       move_to_drive=False,
                       move_to_drive_batch=False,
                       cdr=fixed(cdr),
                       verbose=fixed(True))

#############################################################################
#############################################################################
#############################################################################
##################### Low continuous ##############


def computation_continuousLow(df,
                              dic_df,
                              index_continuous,
                              index_cat,
                              log,
                              sample=False):
    """
    Compute the following:

    - Anova


    """

    var_continuous = dic_df['var_continuous'][index_continuous]['variable_db']
    name_continuous = dic_df['var_continuous'][index_continuous]['name']

    drop_continuous = dic_df['var_continuous'][index_continuous]['drop_value']

    drop_decile = dic_df['var_continuous'][index_continuous]['drop_decile']

    var_categorical = dic_df['var_categorical_low'][index_cat]['variable_db']
    name_categorical = dic_df['var_categorical_low'][index_cat]['name']

    drop_categorical = dic_df['var_categorical_low'][index_cat]['Drop']

    df_var = df[[var_continuous, var_categorical]]

    # Drop if needed

    # Remove decilce

    # Can drop Year or categorical

    if len(drop_continuous) > 0:
        # print(drop_continuous)
        df_var = df_var.loc[~df_var.isin(drop_continuous).any(1)]

    if len(drop_decile) > 0:
        # print(drop_decile)
        lower_d = drop_decile[0]
        value_lower_d = df_var[var_continuous].quantile(lower_d)

        higher_d = drop_decile[1]
        value_higher_d = df_var[var_continuous].quantile(higher_d)

        df_var = df_var[df_var[var_continuous] > value_lower_d]
        df_var = df_var[df_var[var_continuous] < value_higher_d]

        df_decile = pd.DataFrame({
            'Decile': [lower_d, higher_d],
            'Value': [value_lower_d, value_higher_d]
        })
    else:
        df_decile = pd.DataFrame({
            'Decile': [0, 1],
            'Value': [np.min(df_var[var_continuous]),
                      np.max(df_var[var_continuous])]
        })

    # Can drop Year or categorical

    if len(drop_categorical) > 0:
        print(drop_categorical)
        df_var = df_var.loc[~df_var.isin(drop_categorical).any(1)]

    # return log continuous if log
    if log:
        df_1 = df_var.copy()
        df_1[var_continuous] = np.log(df_1[var_continuous] + 1)
    else:
        df_1 = df_var.copy()

    # Prepare var to compute density and loop over each group in cat
    max_y = np.max(df_1[var_continuous])
    min_y = np.min(df_1[var_continuous])
    mean_y = np.mean(df_1[var_continuous])
    sd_y = np.std(df_1[var_continuous])
    #hist_data = []
    group_label = []
    unique_group = df_1[var_categorical].unique()

    x_cdf = np.linspace(mean_y - 3 * sd_y, mean_y + 3 * sd_y, 100)

    df_density = pd.DataFrame({'x': x_cdf})

    # Append each group in list
    for x, name in enumerate(unique_group):
        #serie = "x_" + str(name)
        serie = df_1[df_1[var_categorical] == name][var_continuous]
        # hist_data.append(serie)
        group_label.append(name)
        # print(serie)

        # Compute cdf/pdf

        mean = np.mean(serie)
        sd = np.std(serie)
        med = np.median(serie)

        name_cdf = 'cdf' + str(name)
        name_pdf = 'pdf' + str(name)

        y_cdf = pd.Series(ss.norm.cdf(x_cdf, mean, sd), name=name_cdf)
        y_pdf = pd.Series(ss.norm.pdf(x_cdf, mean, sd), name=name_pdf)

        df_density = pd.concat([df_density, y_cdf, y_pdf], axis=1)

    # Compute summary statistic
    # rp.summary returns a dataframe
    sum_y = rp.summary_cont(df_1[var_continuous])
    sum_y_group = rp.summary_cont(df_1[var_continuous].groupby(
        df_1[var_categorical])).reset_index()

    if sample != False:
        df_1 = df_1.sample(frac=sample, replace=True)

    # Compute ANOVA
    #df = sample_data(df, sample)
    len_g = df_1[var_categorical].unique()
    temp = df_1[[var_continuous, var_categorical]]
    result = temp.groupby(var_categorical)[var_continuous].apply(list)
    F, p = stats.f_oneway(*result)
    r_s = ('Stat=%.3f, p=%.3f' % (F, p))
    alpha = 0.05
    if p > alpha:
        test = 'Not different (H0)'
    else:
        test = 'Different (No H0)'
    l_resut = [[test, F, p]]

    # Anova Turkey test
    mc = MultiComparison(df_1[var_continuous], df_1[var_categorical])
    mc_results = mc.tukeyhsd()
    df_mc = pd.read_html(
        mc_results.summary().as_html(), header=0,
        index_col=0)[0].reset_index()

    dic_int = {
        'output': [
            group_label, df_density, df_mc, sum_y, df_decile, sum_y_group,
            l_resut, var_categorical, name_continuous, name_categorical
        ]
    }

    return dic_int


def saveToDriveLow(cdr=False,
                   sheetID=False,
                   sheetName=False,
                   nb_group=False,
                   LatestRow=False,
                   df_density=False,
                   df_mc=False,
                   sum_y=False,
                   sum_y_group=False,
                   df_decile=False,
                   l_resut=False
                   ):
    """
    """

    nb_cols = df_density.shape[1]
    n_rows = df_density.shape[0]
    n_row_tukey = df_mc.shape[0]
    begin = LatestRow + 4
    # begin = end_row
    # n_end_row = begin + len_btw + nb_rows + nb_group

    # get range for Google Sheet
    # alphabet = ['A', 'B', 'C', 'D','E','F','G','H','I','J','K','L','M','N','O',
    #            'P']

    for i, letter in enumerate(alphabet):
        if i == nb_cols:
            range_1_letter = letter

    dic_range = {

        'range_summary': "A" + str(begin) + ":G" + str(begin + 1),
        'range_remove': "A" + str(begin + 2) + ":C" + str(begin + 4),
        'range_summary_group': "I" + str(begin) + ":O" + str(begin + nb_group),
        'range_anova': "Q" + str(begin) + ":W" + str(begin + 3),
        # 'range_density': "A" + str(begin + nb_group + 5) + ":" +
        # str(range_1_letter) + str(begin + nb_group + 5 + n_rows),
        'range_tukey': "U" + str(begin) + ":Z" + str(begin + n_row_tukey + 1),
        # 'last_row': n_end_row
    }

    table_output = {
        'summary': sum_y.to_numpy().tolist(),
        'drop_decile':   df_decile.to_numpy().tolist(),
        'summary_group': sum_y_group.to_numpy().tolist(),
        'result_test': l_resut,
        # 'table_density': df_density.to_numpy().tolist(),
        'resultat_tukey': df_mc.to_numpy().tolist(),
        'ranges': dic_range,
        'header_summary': list(sum_y),
        'header_decile': list(df_decile),
        'header_summary_group': list(sum_y_group),
        'header_anova': ['result', 'F-test', 'p_value'],
        # 'header_density': list(df_density),
        'header_tukey': list(df_mc)
    }

    # Summary data
    cdr.add_data_to_spreadsheet(data=table_output['summary'],
                                sheetID=sheetID,
                                sheetName=sheetName,
                                rangeData=table_output['ranges']['range_summary'],
                                headers=table_output['header_summary'])

    # decile data
    cdr.add_data_to_spreadsheet(data=table_output['drop_decile'],
                                sheetID=sheetID,
                                sheetName=sheetName,
                                rangeData=table_output['ranges']['range_remove'],
                                headers=table_output['header_decile'])

    # Summary group data
    cdr.add_data_to_spreadsheet(data=table_output['summary_group'],
                                sheetID=sheetID,
                                sheetName=sheetName,
                                rangeData=table_output['ranges']['range_summary_group'],
                                headers=table_output['header_summary_group'])

    # Anova
    cdr.add_data_to_spreadsheet(data=table_output['result_test'],
                                sheetID=sheetID,
                                sheetName=sheetName,
                                rangeData=table_output['ranges']['range_anova'],
                                headers=table_output['header_anova'])

    cdr.add_data_to_spreadsheet(data=table_output['resultat_tukey'],
                                sheetID=sheetID,
                                sheetName=sheetName,
                                rangeData=table_output['ranges']['range_tukey'],
                                headers=table_output['header_tukey'])


def summary_continuous_low_dimension(df,
                                     dic_multiple,
                                     log=False,
                                     sample=False,
                                     variables=False,
                                     sheetID=False,
                                     sheetName=False,
                                     move_to_drive=False,
                                     move_to_drive_batch=False,
                                     cdr=False,
                                     verbose=True):
    """
    Variables needs to be a 2d array
    """

    if move_to_drive_batch:

        max_ = len(dic_multiple['var_continuous']) * \
            len(dic_multiple['var_categorical_low'])
        # instantiate the bar
        f = IntProgress(min=0, max=max_ + 1, description='Loading:')
        display(f)

        for key, value in dic_multiple['var_continuous'].items():
            variable_c = value['name']

            f.value += 1

            for key, value in dic_multiple['var_categorical_low'].items():
                variable_cat = value['name']

                f.value += 1

                for key, value in dic_multiple['var_continuous'].items():
                    if variable_c == value['name']:
                        index_dic_g1 = key

                for key, value in dic_multiple['var_categorical_low'].items():
                    if variable_cat == value['name']:
                        index_dic_g2 = key
    # index_dic = [index_dic_g1, index_dic_g2]

                index_continuous = index_dic_g1
                index_cat = index_dic_g2

                temp_comp = computation_continuousLow(df=df, dic_df=dic_multiple,
                                                      index_continuous=index_continuous,
                                                      index_cat=index_cat,
                                                      log=log,
                                                      sample=False)

    # range
    # summary

                LatestRow = cdr.getLatestRow(
                    sheetID=sheetID, sheetName=sheetName)

    # Extract from temp_comp

                group_label = temp_comp['output'][0]
                df_density = temp_comp['output'][1]
                df_mc = temp_comp['output'][2]
                sum_y = temp_comp['output'][3]
                df_decile = temp_comp['output'][4]
                sum_y_group = temp_comp['output'][5]
                l_resut = temp_comp['output'][6]
                nb_group = len(group_label)

                saveToDriveLow(cdr=cdr,
                               sheetID=sheetID,
                               sheetName=sheetName,
                               nb_group=nb_group,
                               LatestRow=LatestRow,
                               df_density=df_density,
                               df_mc=df_mc,
                               sum_y=sum_y,
                               sum_y_group=sum_y_group,
                               df_decile=df_decile,
                               l_resut=l_resut
                               )

    # len_btw = 7

        f.value += 1
    if verbose:
        # TEST widgets
        # Define output

        regex = r"^[^-]+"
        regex_1 = r"\-(.*)"
        g1 = re.search(regex, variables)[0]
        g2 = re.search(regex_1, variables)[1]

        for key, value in dic_multiple['var_continuous'].items():
            if g1 == value['name']:
                index_dic_g1 = key

        for key, value in dic_multiple['var_categorical_low'].items():
            if g2 == value['name']:
                index_dic_g2 = key

        temp_comp = computation_continuousLow(df=df,
                                              dic_df=dic_multiple,
                                              index_continuous=index_dic_g1,
                                              index_cat=index_dic_g2,
                                              log=log,
                                              sample=False)
        group_label = temp_comp['output'][0]
        df_density = temp_comp['output'][1]
        df_mc = temp_comp['output'][2]
        sum_y = temp_comp['output'][3]
        df_decile = temp_comp['output'][4]
        sum_y_group = temp_comp['output'][5]
        var_categorical = temp_comp['output'][7]
        l_resut = temp_comp['output'][6]
        name_continuous = temp_comp['output'][8]
        name_categorical = temp_comp['output'][9]
        nb_group = len(group_label)

        if move_to_drive:
            LatestRow = cdr.getLatestRow(
                sheetID=sheetID, sheetName=sheetName)

            saveToDriveLow(cdr=cdr,
                           sheetID=sheetID,
                           sheetName=sheetName,
                           nb_group=nb_group,
                           LatestRow=LatestRow,
                           df_density=df_density,
                           df_mc=df_mc,
                           sum_y=sum_y,
                           sum_y_group=sum_y_group,
                           df_decile=df_decile,
                           l_resut=l_resut
                           )

        # cf.go_offline()
        #cf.set_config_file(offline=False, world_readable=False, theme='space')

        summary_ = widgets.Output()
        summary_tukey = widgets.Output()
        summary_plot = widgets.Output()

        tab_contents = [summary_plot, summary_, summary_tukey]
        tab = widgets.Tab(tab_contents)
        tab.set_title(0, 'Distribution')
        tab.set_title(1, 'Summary Statistic')
        tab.set_title(2, 'Tukey Results')

        display(tab)

        with summary_:
            temp = pd.concat([sum_y,
                              sum_y_group.rename(index=str,
                                                 columns={var_categorical:
                                                          "Variable"})])

            #temp = temp.style.format({'SD': "{:.2}", 'SE': '±{:.2f}'})
            #cm = sns.light_palette("green", as_cmap=True)

            # temp_1 = temp.style.bar(subset=['Mean'],
            #    align='mid',
            #                      color=['#d65f5f', '#5fba7d'])
            temp = temp.style.format({'SD': "{:.2}", 'SE': '±{:.2f}'})
            #temp = temp.style.format({'SE': '±{:.2f}'})
            #['Variable', 'N', 'Mean', 'SD', 'SE', '95% Conf.', 'Interval']
            display(temp)
        with summary_tukey:
            # Color cell
            # df_mc = df_mc.style.apply(highlight_max, subset=['meandiff'])
            df_mc = df_mc.style.bar(subset=['meandiff'],
                                    align='mid',
                                    color=['#d65f5f', '#5fba7d'])

            display(df_mc)

        with summary_plot:
            #f, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=False)

            for i, dist in enumerate(['cdf', 'pdf']):
                filter_col = [
                    col for col in df_density if col.startswith(dist)]
                filter_col.append('x')
                df_cdf = df_density[filter_col]
                df_cdf = df_cdf.set_index('x').stack().reset_index()
                df_cdf.columns = ['x', 'group', 'value']

                fig = px.line(df_cdf,
                              x='x',
                              y='value',
                              color='group',
                              template="plotly_dark")
                fig.update_traces(textposition='top center')
                fig.update_layout(
                    height=800,
                    title_text='Density distribution of ' + name_continuous +
                    ' grouped by ' + name_categorical)

                fig.show()

                # df_cdf.iplot(subplots=False,
                #				yTitle=name_continuous,
                #				 title= 'Density distribution of ' + name_continuous +
                #			' grouped by '+ name_categorical)

                # print(df_cdf)

            #   for y in filter_col[:-1]:
            #        ax = sns.lineplot(x='x',
            #                          y=y,
            #                          data=df_cdf,
            #                          label=y,
            #                          ax=axes[i]
            #                          )
            # plt.show()


def list_dropdownLow(dic_df):
    """
    """

    l_choice = []
    for key_cont, value_cont in dic_df['var_continuous'].items():

        for key_cat, value_cat in dic_df['var_categorical_low'].items():
            #l_choice.append([value_cont['name'], value_cat['name']])
            l_choice.append(value_cont['name'] + '-' + value_cat['name'])
    l_choice = ['-'] + l_choice

    return l_choice


def select_catLow_eventHandler(df, dic_df, cdr=False):
    """
    """

    x_widget = widgets.Dropdown(
        options=list_dropdownLow(dic_df=dic_df),
        value='-',
        description='Variable:')

    return interactive(summary_continuous_low_dimension,
                       {"manual": True, "auto_display": False},
                       df=fixed(df),
                       dic_multiple=fixed(dic_df),
                       log=False,
                       sample=fixed(False),
                       variables=x_widget,
                       sheetID='',
                       sheetName='',
                       move_to_drive=False,
                       move_to_drive_batch=False,
                       cdr=fixed(cdr),
                       verbose=fixed(True))

#############################################################################
#############################################################################
#############################################################################
##################### High continuous ##############


def computation_continuousHigh(df,
                               dic_df,
                               index_continuous,
                               index_cat,
                               var_cat_color,
                               log,
                               sample=False):
    """
    var_cat_color is the variable to add color:
    Can be a var name or False
    """

    var_continuous = dic_df['var_continuous'][index_continuous][
        'variable_db']
    name_continuous = dic_df['var_continuous'][index_continuous]['name']

    drop_continuous = dic_df['var_continuous'][index_continuous][
        'drop_value']

    drop_decile = dic_df['var_continuous'][index_continuous][
        'drop_decile']

    var_categorical = dic_df['var_categorical_high'][index_cat][
        'variable_db']
    name_categorical = dic_df['var_categorical_high'][index_cat]['name']

    drop_categorical = dic_df['var_categorical_high'][index_cat]['Drop']

    #var_cat_color = dic_df['var_categorical_high'][index_cat]['color']

    if var_cat_color != False:
        df_var = df[[var_continuous, var_categorical, var_cat_color]]
    else:
        df_var = df[[var_continuous, var_categorical]]

    # Drop if needed

    if len(drop_continuous) > 0:
        # print(drop_continuous)
        df_var = df_var.loc[~df_var.isin(drop_continuous).any(1)]

    if len(drop_decile) > 0:
        # print(drop_decile)
        lower_d = drop_decile[0]
        value_lower_d = df_var[var_continuous].quantile(lower_d)

        higher_d = drop_decile[1]
        value_higher_d = df_var[var_continuous].quantile(higher_d)

        df_var = df_var[df_var[var_continuous] > value_lower_d]
        df_var = df_var[df_var[var_continuous] < value_higher_d]

        df_decile = pd.DataFrame({
            'Decile': [lower_d, higher_d],
            'Value': [value_lower_d, value_higher_d]
        })
    else:
        df_decile = pd.DataFrame({
            'Decile': [0, 1],
            'Value': [np.min(df_var[var_continuous]),
                      np.max(df_var[var_continuous])]
        })

    # Can drop Year or categorical

    if len(drop_categorical) > 0:
        # print(drop_categorical)
        df_var = df_var.loc[~df_var.isin(drop_categorical).any(1)]

    # return log continuous if log
    if log:
        df_1 = df_var.copy()
        df_1[var_continuous] = np.log(df_1[var_continuous] + 1)
    else:
        df_1 = df_var.copy()

    # Prepare var to compute density and loop over each group in cat
    max_y = np.max(df_1[var_continuous])
    hist_data = []
    group_label = []
    unique_group = df_1[var_categorical].unique()

    # Append each group in list
    for x, name in enumerate(unique_group):
        serie = "x_" + str(name)
        serie = df_1[df_1[var_categorical] == name][var_continuous]
        hist_data.append(serie)
        group_label.append(name)

    # Compute summary statistic
    # rp.summary returns a dataframe
    sum_y = rp.summary_cont(df_1[var_continuous])

    if var_cat_color != False:

        sum_y_group = rp.summary_cont(df_1.groupby(
            [var_categorical, var_cat_color])[var_continuous]
        ).reset_index()
    else:
        sum_y_group = rp.summary_cont(df_1[var_continuous].groupby(
            df_1[var_categorical])).reset_index().sort_values(by='Mean')

    # print(var_cat_color)

    # if var_cat_color != False:
    #	df_color = df[[var_categorical, var_cat_color]]
    #	df_color = df_color.drop_duplicates()

    #	sum_y_group = pd.merge(
    #		sum_y_group, df_color, on=var_categorical, how='left')

    if sample != False:
        df_1 = df_1.sample(frac=sample, replace=True)

    # Compute ANOVA
    #df = sample_data(df, sample)
    len_g = df_1[var_categorical].unique()
    temp = df_1[[var_continuous, var_categorical]]
    result = temp.groupby(var_categorical)[var_continuous].apply(list)
    F, p = stats.f_oneway(*result)
    r_s = ('Stat=%.3f, p=%.3f' % (F, p))
    alpha = 0.05
    if p > alpha:
        test = 'Not different (H0)'
    else:
        test = 'Different (No H0)'
    l_resut = [[test, F, p]]

    # Anova Turkey test
    mc = MultiComparison(df_1[var_continuous], df_1[var_categorical])
    mc_results = mc.tukeyhsd()
    df_mc = pd.read_html(
        mc_results.summary().as_html(), header=0,
        index_col=0)[0].reset_index()

    dic_int = {
        'output': [
            group_label, df_mc, sum_y, df_decile, sum_y_group,
            l_resut, var_categorical, name_continuous, name_categorical,
            var_cat_color
        ]
    }

    return dic_int


def saveToDriveHigh(cdr=False,
                    sheetID=False,
                    sheetName=False,
                    LatestRow=False,
                    group_label=False,
                    nb_group=False,
                    var_cat_color=False,
                    df_mc=False,
                    sum_y=False,
                    sum_y_group=False,
                    l_resut=False
                    ):

    len_btw = 5
    nb_group = len(group_label)
    nb_cols = df_mc.shape[1]
    n_rows = df_mc.shape[0]
    nb_rows = n_rows + 1
    begin = LatestRow + 4

    if var_cat_color != False:
        nb_group = sum_y_group.shape[0]
        sum_y_group = sum_y_group.fillna('')

    # get range for Google Sheet
    # alphabet = ['A', 'B', 'C', 'D','E','F','G','H','I','J','K','L','M','N','O',
    #            'P']

    for i, letter in enumerate(alphabet):
        if i == nb_cols:
            range_1_letter = letter

    dic_range = {
        'range_summary':
            "A" + str(begin) + ":G" + str(begin + 1),
            'range_summary_group':
            "I" + str(begin) + ":P" + str(begin + nb_group),
            'range_anova':
            "R" + str(begin) + ":X" + str(begin + 3),
            'range_tukey':
            "W" + str(begin) + ":AB1" + str(begin + n_rows + 1),
            # 'last_row': n_end_row
    }

    table_output = {
        'summary': sum_y.to_numpy().tolist(),
        'summary_group': sum_y_group.to_numpy().tolist(),
        'result_test': l_resut,
        'resultat_tukey': df_mc.to_numpy().tolist(),
        'ranges': dic_range,
        'header_summary': list(sum_y),
        'header_summary_group': list(sum_y_group),
        'header_anova': ['result', 'F-test', 'p_value'],
        'header_tukey': list(df_mc)
    }

    cdr.add_data_to_spreadsheet(
        data=table_output['summary'],
        sheetID=sheetID,
        sheetName=sheetName,
        rangeData=table_output['ranges']['range_summary'],
        headers=table_output['header_summary'])

    # Summary group data
    cdr.add_data_to_spreadsheet(
        data=table_output['summary_group'],
        sheetID=sheetID,
        sheetName=sheetName,
        rangeData=table_output['ranges']['range_summary_group'],
        headers=table_output['header_summary_group'])

    # Anova
    cdr.add_data_to_spreadsheet(
        data=table_output['result_test'],
        sheetID=sheetID,
        sheetName=sheetName,
        rangeData=table_output['ranges']['range_anova'],
        headers=table_output['header_anova'])

    # Tukey
    # Anova
    cdr.add_data_to_spreadsheet(
        data=table_output['resultat_tukey'],
        sheetID=sheetID,
        sheetName=sheetName,
        rangeData=table_output['ranges']['range_tukey'],
        headers=table_output['header_tukey'])


def summary_continuous_high_dimension(df,
                                      dic_multiple,
                                      log=False,
                                      sample=False,
                                      variables=False,
                                      var_cat_color=False,
                                      sheetID=False,
                                      sheetName=False,
                                      move_to_drive=False,
                                      move_to_drive_batch=False,
                                      cdr=False,
                                      verbose=False):
    """
    Add ANOVA Tukey HSD test
    """

    if move_to_drive_batch:

        max_ = len(dic_multiple['var_continuous']) + \
            len(dic_multiple['var_categorical_high'])
        # instantiate the bar
        f = IntProgress(min=0, max=max_ + 1, description='Loading:')
        display(f)

        for key, value in dic_multiple['var_continuous'].items():
            variable_c = value['name']

            f.value += 1

            for key, value in dic_multiple['var_categorical_high'].items():
                variable_cat = value['name']

                f.value += 1

                for key, value in dic_multiple['var_continuous'].items():
                    if variable_c == value['name']:
                        index_dic_g1 = key

                for key, value in dic_multiple['var_categorical_high'].items():
                    if variable_cat == value['name']:
                        index_dic_g2 = key
    # index_dic = [index_dic_g1, index_dic_g2]

                index_continuous = index_dic_g1
                index_cat = index_dic_g2
                # Extract each component of the dictionary

                temp_comp = computation_continuousHigh(
                    df=df,
                    dic_df=dic_multiple,
                    index_continuous=index_continuous,
                    index_cat=index_cat,
                    var_cat_color=False,
                    log=log,
                    sample=False)

                group_label = temp_comp['output'][0]
                df_mc = temp_comp['output'][1]
                sum_y = temp_comp['output'][2]
                df_decile = temp_comp['output'][3]
                sum_y_group = temp_comp['output'][4]
                l_resut = temp_comp['output'][5]

                LatestRow = cdr.getLatestRow(
                    sheetID=sheetID, sheetName=sheetName)

                saveToDriveHigh(cdr=cdr,
                                sheetID=sheetID,
                                sheetName=sheetName,
                                LatestRow=LatestRow,
                                group_label=group_label,
                                nb_group=group_label,
                                var_cat_color=var_cat_color,
                                df_mc=df_mc,
                                sum_y=sum_y,
                                sum_y_group=sum_y_group,
                                l_resut=l_resut
                                )

        f.value += 1
    if verbose:

        regex = r"^[^-]+"
        regex_1 = r"\-(.*)"
        g1 = re.search(regex, variables)[0]
        g2 = re.search(regex_1, variables)[1]

        for key, value in dic_multiple['var_continuous'].items():
            if g1 == value['name']:
                index_dic_g1 = key

        for key, value in dic_multiple['var_categorical_high'].items():
            if g2 == value['name']:
                index_dic_g2 = key
    # index_dic = [index_dic_g1, index_dic_g2]

        index_continuous = index_dic_g1
        index_cat = index_dic_g2
        # Extract each component of the dictionary

        temp_comp = computation_continuousHigh(
            df=df,
            dic_df=dic_multiple,
            index_continuous=index_continuous,
            index_cat=index_cat,
            var_cat_color=var_cat_color,
            log=log,
            sample=False)

        group_label = temp_comp['output'][0]
        df_mc = temp_comp['output'][1]
        sum_y = temp_comp['output'][2]
        sum_y_group = temp_comp['output'][4]
        l_resut = temp_comp['output'][5]
        var_categorical = temp_comp['output'][6]
        name_continuous = temp_comp['output'][7]
        name_categorical = temp_comp['output'][8]
        ####
        var_cat_color = temp_comp['output'][9]

        #df_var = temp_comp['output'][10]

        if move_to_drive:

            LatestRow = cdr.getLatestRow(
                sheetID=sheetID, sheetName=sheetName)

            saveToDriveHigh(cdr=cdr,
                            sheetID=sheetID,
                            sheetName=sheetName,
                            LatestRow=LatestRow,
                            group_label=group_label,
                            nb_group=group_label,
                            var_cat_color=var_cat_color,
                            df_mc=df_mc,
                            sum_y=sum_y,
                            sum_y_group=sum_y_group,
                            l_resut=l_resut
                            )

        summary_ = widgets.Output()
        summary_bar_polar = widgets.Output()
        summary_tukey = widgets.Output()
        summary_heatmap = widgets.Output()
        summary_true = widgets.Output()

        tab_contents = [summary_bar_polar, summary_, summary_tukey,
                        summary_heatmap, summary_true]
        tab = widgets.Tab(tab_contents)
        tab.set_title(0, 'Summary Bar polar')
        tab.set_title(1, 'Summary Statistic')
        tab.set_title(2, 'Tukey Results')
        tab.set_title(3, 'Heatmap')
        tab.set_title(4, 'True Only')

        display(tab)

        with summary_:

            # temp = pd.concat([
            # sum_y,
                 #   sum_y_group.rename(
                 #       index=str, columns={var_categorical: "Variable"})
            # ]).loc
            # change color cells by griup

            if var_cat_color != False:
                unique_color = df[var_cat_color].unique()
                unique_group_c = setcolors(unique_color)

                print('the global average is {0}'.format(sum_y['Mean']))
                temp = sum_y_group.rename(
                    index=str, columns={var_categorical: name_categorical})
                cm = sns.light_palette("green", as_cmap=True)
                temp = (temp.style
                        .bar(subset=['Mean', 'N'], align='mid',
                             color=['#d65f5f', '#5fba7d'])
                        .format({
                            'SE': '±{:.2f}'})
                        .applymap(applycolors,
                                  l_colors=unique_group_c,
                                  subset=[var_cat_color]))
            else:
                temp = sum_y_group.rename(
                    index=str, columns={var_categorical: name_categorical})
                cm = sns.light_palette("green", as_cmap=True)
                temp = (temp.style
                        .bar(subset=['Mean', 'N'], align='mid',
                             color=['#d65f5f', '#5fba7d'])
                        .format({
                            'SE': '±{:.2f}'})
                        )

            display(temp)

        with summary_bar_polar:
            if var_cat_color != False:

                fig = px.bar_polar(sum_y_group,
                                   r="Mean",
                                   theta=var_categorical,
                                   color=var_cat_color,
                                   template="plotly_dark",
                                   color_discrete_sequence=px.colors.sequential.Plasma[-2::-1])

                fig.show()

                fig_1 = px.bar(sum_y_group,
                               x=var_categorical,
                               y='Mean',
                               color=var_cat_color)

                fig_1.show()

        with summary_tukey:
            # Color cell
            # df_mc = df_mc.style.bar(subset=['meandiff'],
            #                        align='mid',
            #                        color=['#d65f5f', '#5fba7d'])
            df_mc_t = df_mc.pivot(
                index='group1', columns='group2', values='reject')
            df_mc_t = df_mc_t.fillna('-')
            # cm = sns.light_palette("green", as_cmap=True)
            s = df_mc_t.style.apply(highlight_reject)
            display(s)

        with summary_heatmap:
            df_mc_h = df_mc.pivot(
                index='group1', columns='group2', values='meandiff')
            df_mc_h = df_mc_h.fillna(0)

            cm = sns.light_palette("red", as_cmap=True)

            s1 = df_mc_h.style.background_gradient(cmap='viridis')
            # s1 = df_mc_h.style.bar(color='#d65f5f')
            display(s1)

        with summary_true:
            df_mc_t = df_mc[df_mc['reject'] == True]
            df_mc_t = (df_mc_t.style
                       .bar(subset=['meandiff'], align='mid',
                            color=['#d65f5f', '#5fba7d'])
                       .format({
                           'SE': '±{:.2f}'})
                       )

            display(df_mc_t)


def list_dropdownHigh(dic_df):
    """
    """

    l_choice = []
    for key_cont, value_cont in dic_df['var_continuous'].items():

        for key_cat, value_cat in dic_df['var_categorical_high'].items():
            l_choice.append(value_cont['name'] + '-' + value_cat['name'])

    l_choice = ['-'] + l_choice

    return l_choice


def select_catHigh_eventHandler(df, dic_df, cdr=False):
    """
    """
    l_cat = list(df.select_dtypes(include='object'))
    l_cat = [False] + l_cat

    x_widget = widgets.Dropdown(
        options=list_dropdownHigh(dic_df=dic_df),
        value='-',
        description='Variable:')

    return interactive(summary_continuous_high_dimension,
                       {"manual": True, "auto_display": False},
                       df=fixed(df),
                       dic_multiple=fixed(dic_df),
                       log=False,
                       sample=fixed(False),
                       variables=x_widget,
                       var_cat_color=l_cat,
                       sheetID='',
                       sheetName='',
                       move_to_drive=False,
                       move_to_drive_batch=False,
                       cdr=fixed(cdr),
                       verbose=fixed(True))


#############################################################################
#############################################################################
#############################################################################
##################### High/Low continuous ##############
def computation_slopeRank(df,
                          dic_df,
                          index_continuous,
                          index_cat_high,
                          index_cat_low,
                          log,
                          sample=False):
    """
    """

    # Extract each component of the dictionary

    var_continuous = dic_df['var_continuous'][index_continuous]['variable_db']
    name_continuous = dic_df['var_continuous'][index_continuous]['name']

    drop_continuous = dic_df['var_continuous'][index_continuous]['drop_value']

    drop_decile = dic_df['var_continuous'][index_continuous]['drop_decile']

    var_categorical_high = dic_df['var_categorical_high'][index_cat_high][
        'variable_db']
    name_categorical_high = dic_df['var_categorical_high'][index_cat_high][
        'name']

    drop_categorical_high = dic_df['var_categorical_high'][index_cat_high][
        'Drop']

    var_categorical_low = dic_df['var_categorical_low'][index_cat_low][
        'variable_db']
    name_categorical_low = dic_df['var_categorical_low'][index_cat_low][
        'name']

    drop_categorical_low = dic_df['var_categorical_low'][index_cat_low]['Drop']

    df_var = df[[var_continuous, var_categorical_high, var_categorical_low]]

    # Drop if needed

    if len(drop_continuous) > 0:
        # print(drop_continuous)
        df_var = df_var.loc[~df_var.isin(drop_continuous).any(1)]

    if len(drop_decile) > 0:
        # print(drop_decile)
        lower_d = drop_decile[0]
        value_lower_d = df_var[var_continuous].quantile(lower_d)

        higher_d = drop_decile[1]
        value_higher_d = df_var[var_continuous].quantile(higher_d)

        df_var = df_var[df_var[var_continuous] > value_lower_d]
        df_var = df_var[df_var[var_continuous] < value_higher_d]

        df_decile = pd.DataFrame({
            'Decile': [lower_d, higher_d],
            'Value': [value_lower_d, value_higher_d]
        })
    # Can drop Year or categorical

    if len(drop_categorical_high) > 0:
        # print(drop_categorical_high)
        df_var = df_var.loc[~df_var.isin(drop_categorical_high).any(1)]

    if len(drop_categorical_low) > 0:
        # print(var_categorical_low)
        df_var = df_var.loc[~df_var.isin(drop_categorical_low).any(1)]

    # return log continuous if log
    if log:
        df_1 = df_var.copy()
        df_1[var_continuous] = np.log(df_1[var_continuous] + 1)
    else:
        df_1 = df_var.copy()

    classes = df_1[var_categorical_low].unique()

    mean_slope = df_1.groupby(
        [var_categorical_high,
         var_categorical_low])[var_continuous].mean().reset_index().pivot(
        index=var_categorical_high,
        columns=var_categorical_low,
        values=var_continuous).sort_values(
        by=classes[0], ascending=False).reset_index()

    median_slope = df_1.groupby(
        [var_categorical_high,
         var_categorical_low])[var_continuous].median().reset_index().pivot(
        index=var_categorical_high,
        columns=var_categorical_low,
        values=var_continuous).sort_values(
        by=classes[0], ascending=False).reset_index()

    sum_slope = df_1.groupby(
        [var_categorical_high,
         var_categorical_low])[var_continuous].sum().reset_index().pivot(
        index=var_categorical_high,
        columns=var_categorical_low,
        values=var_continuous).sort_values(
        by=classes[0], ascending=False)

    perc_slope = sum_slope.apply(lambda x: x / x.sum()).reset_index()

    df_slope = pd.merge(
        mean_slope,
        median_slope,
        on=var_categorical_high,
        how='left',
        suffixes=('_mean', '_median'))

    df_slope = pd.merge(
        df_slope, sum_slope, on=var_categorical_high, how='left')

    df_slope = pd.merge(
        df_slope,
        perc_slope,
        on=var_categorical_high,
        how='left',
        suffixes=('_sum', '_per'))

    df_slope = df_slope.fillna('')

    dic_int = {
        'output': [
            df_slope, name_continuous, name_categorical_high,
            name_categorical_low, df_var, var_categorical_low,
            df_slope, var_continuous, var_categorical_high
        ]
    }

    return dic_int


def createfigurePlot(df, var_categorical_low, name_categorical_high,
                     df_slope, var_continuous, stat):
    """
    Create a figure to avoid inline plot
    df is df_var
    """

    if len(df[var_categorical_low].unique()) == 2:
        l_var = list(df_slope)
        # for i, stat in enumerate(['mean', 'median', 'sum', 'per']):
        filter_col = [col for col in df_slope if col.endswith(stat)]
        filter_col.append(l_var[0])
        df_slop_filter = df_slope[filter_col]
        df_slop_filter[stat + '_dif'] = df_slop_filter.iloc[:, 1] - \
            df_slop_filter.iloc[:, 0]
        df_slop_filter['colors'] = [
            'red' if x < 0 else 'green'
            for x in df_slop_filter[stat + '_dif']
        ]

        df_slop_filter.sort_values(stat + '_dif', inplace=True)
        df_slop_filter.reset_index(inplace=True)
        # Draw plot
        myFig = plt.figure(figsize=(14, 10), dpi=80)
        plt.hlines(
            y=df_slop_filter.index,
            xmin=0,
            xmax=df_slop_filter[stat + '_dif'],
            color=df_slop_filter.colors,
            alpha=0.4,
            linewidth=5)

        # Decorations
        plt.gca().set(ylabel=var_continuous, xlabel=stat)
        plt.yticks(
            df_slop_filter.index, df_slop_filter.iloc[:, 3], fontsize=12)
        plt.title(
            'Diverging Bars of ' + var_continuous + ' within ' +
            name_categorical_high,
            fontdict={'size': 20})
        plt.grid(linestyle='--', alpha=0.5)

        return myFig


def saveToDriveRank(cdr=False,
                    sheetID=False,
                    sheetName=False,
                    LatestRow=False,
                    folder=False,
                    df_slope=False,
                    df_var=False,
                    var_categorical_low=False,
                    name_categorical_high=False,
                    var_continuous=False
                    ):
    """
    """
    nb_cols = df_slope.shape[1]
    n_rows = df_slope.shape[0]
    begin = LatestRow + 4
    end = begin + n_rows + 1

    for i, letter in enumerate(alphabet):
        if i == nb_cols:
            range_1_letter = letter

    range_slope = "A" + str(begin) + ":" + \
        str(range_1_letter) + str(end)
    table_output = {
        'df_slope': df_slope.to_numpy().tolist(),
        'range_slope': range_slope,
        'header_slope': list(df_slope)
    }
    # return table_output
    # Summary data
    cdr.add_data_to_spreadsheet(
        data=table_output['df_slope'],
        sheetID=sheetID,
        sheetName=sheetName,
        rangeData=table_output['range_slope'],
        headers=table_output['header_slope'])

    for i, stat in enumerate(['mean', 'median', 'sum', 'per']):
        fig_ = createfigurePlot(
            df=df_var,
            var_categorical_low=var_categorical_low,
            name_categorical_high=name_categorical_high,
            df_slope=df_slope,
            var_continuous=var_continuous,
            stat=stat)

        name_ = 'Diverging Bars of ' + var_continuous + ' within ' + \
                name_categorical_high + ' as ' + stat + '.png'
        # There is no plot yet when the low dimensiona group is
        # higher than 2
        fig_.savefig(name_)
        try:

            mime_type = "image/png"
            cdr.upload_file_root(mime_type, name_)
            cdr.move_file(file_name=name_, folder_name=folder)
            os.remove(name_)
        except:
            pass


def slope_rank(df,
               dic_multiple,
               log=False,
               variables=False,
               sheetID=False,
               sheetName=False,
               folder=False,
               move_to_drive=False,
               move_to_drive_batch=False,
               cdr=False,
               verbose=False):
    """
    """

    if move_to_drive_batch:
        max_ = len(dic_multiple['var_continuous']) + \
            len(dic_multiple['var_categorical_high']) + \
            len(dic_multiple['var_categorical_low'])
        # instantiate the bar
        f = IntProgress(min=0, max=max_, description='Loading:')
        display(f)
        # dropdown_1_output.clear_output()
        for key, value in dic_multiple['var_continuous'].items():
            variable_c = value['name']
            f.value += 1
            for key, value in dic_multiple['var_categorical_high'].items():
                variable_cat = value['name']

                for key, value in dic_multiple['var_categorical_low'].items():
                    variable_low = value['name']

                    for key, value in dic_multiple['var_continuous'].items():
                        if variable_c == value['name']:
                            index_dic_g1 = key

                    for key, value in dic_multiple[
                            'var_categorical_high'].items():
                        if variable_cat == value['name']:
                            index_dic_g2 = key

                    for key, value in dic_multiple[
                            'var_categorical_low'].items():
                        if variable_low == value['name']:
                            index_dic_g3 = key

                    index_continuous = index_dic_g1
                    index_cat_h = index_dic_g2
                    index_cat_l = index_dic_g3

                    temp_comp = computation_slopeRank(
                        df=df,
                        dic_df=dic_multiple,
                        index_continuous=index_continuous,
                        index_cat_high=index_cat_h,
                        index_cat_low=index_cat_l,
                        log=log,
                        sample=False)

                    df_slope = temp_comp['output'][0]
                    name_categorical_high = temp_comp['output'][2]
                    df_var = temp_comp['output'][4]
                    var_categorical_low = temp_comp['output'][5]
                    df_slope = temp_comp['output'][6]
                    var_continuous = temp_comp['output'][7]

                    # Extract each component of the dictionary

                    LatestRow = cdr.getLatestRow(
                        sheetID=sheetID, sheetName=sheetName)

                    saveToDriveRank(cdr=cdr,
                                    sheetID=sheetID,
                                    sheetName=sheetName,
                                    LatestRow=LatestRow,
                                    folder=folder,
                                    df_slope=df_slope,
                                    df_var=df_var,
                                    var_categorical_low=var_categorical_low,
                                    name_categorical_high=name_categorical_high,
                                    var_continuous=var_continuous)

                    f.value += 1

    if verbose:
        regex = r"^[^-]+"
        regex_1 = r"\-(.*?)\-"
        regex_2 = r"\-(.*)"
        # Very ugly way
        g1 = re.search(regex, variables)[0]
        g2 = re.search(regex_1, variables)[1]
        g3 = re.search(regex_2, variables)[1]
        g4 = re.search(regex_2, g3)[1]

        for key, value in dic_multiple['var_continuous'].items():
            if g1 == value['name']:
                index_dic_g1 = key

        for key, value in dic_multiple['var_categorical_high'].items():
            if g2 == value['name']:
                index_dic_g2 = key

        for key, value in dic_multiple['var_categorical_low'].items():
            if g4 == value['name']:
                index_dic_g3 = key

        temp_comp = computation_slopeRank(
            df=df,
            dic_df=dic_multiple,
            index_continuous=index_dic_g1,
            index_cat_high=index_dic_g2,
            index_cat_low=index_dic_g3,
            log=log,
            sample=False)

        df_slope = temp_comp['output'][0]
        name_categorical_high = temp_comp['output'][2]
        df_var = temp_comp['output'][4]
        var_categorical_low = temp_comp['output'][5]
        df_slope = temp_comp['output'][6]
        var_continuous = temp_comp['output'][7]
        var_categorical_high = temp_comp['output'][8]

        if move_to_drive:
            LatestRow = cdr.getLatestRow(
                sheetID=sheetID, sheetName=sheetName)

            saveToDriveRank(cdr=cdr,
                            sheetID=sheetID,
                            sheetName=sheetName,
                            LatestRow=LatestRow,
                            folder=folder,
                            df_slope=df_slope,
                            df_var=df_var,
                            var_categorical_low=var_categorical_low,
                            name_categorical_high=name_categorical_high,
                            var_continuous=var_continuous)

        summary_ = widgets.Output()
        summary_plot = widgets.Output()
        summary_rank = widgets.Output()
        #summary_rankPlot= widgets.Output()
        tab_contents = [summary_rank, summary_, summary_plot]
        tab = widgets.Tab(tab_contents)
        tab.set_title(0, 'rank')
        tab.set_title(1, 'Summary Statistic')
        tab.set_title(2, 'Difference plot')
        #tab.set_title(3, 'Difference plot 2')

        display(tab)

        with summary_:
            cm = sns.light_palette("green", as_cmap=True)

            s = df_slope.style.background_gradient(cmap=cm)
            display(s)

        with summary_plot:
            for i, stat in enumerate(['mean', 'median', 'sum', 'per']):
                if len(df_var[var_categorical_low].unique()) > 2:
                    print('This tab plots only graph with group containing \
					2 unique values')
                else:
                    fig_ = createfigurePlot(
                        df=df_var,
                        var_categorical_low=var_categorical_low,
                        name_categorical_high=name_categorical_high,
                        df_slope=df_slope,
                        var_continuous=var_continuous,
                        stat=stat)
                    plt.show()

        with summary_rank:
            classes = np.unique(
                df_var[var_categorical_low].values).tolist()
            parallel_df = df_var.groupby([var_categorical_high,
                                          var_categorical_low]
                                         )[var_continuous].mean().reset_index().pivot(
                index=var_categorical_high,
                columns=var_categorical_low,
                values=var_continuous).sort_values(
                by=classes[0],
                ascending=False).rank(ascending=False)

            size = len(parallel_df)
            dimension = []
            for x, name in enumerate(classes):
                serie = "x_" + str(name)
                serie = parallel_df[name]
                if x == 0:
                    dimension_temp = [dict(range=[1, size],
                                           constraintrange=[0, 10],
                                           tickvals=serie,
                                           ticktext=serie.index,
                                           label=name,
                                           values=serie)]
                else:
                    dimension_temp = [dict(range=[1, size],
                                           tickvals=serie,
                                           ticktext=serie.index,
                                           label=name,
                                           values=serie)]
                dimension.extend(dimension_temp)
            fig = [
                go.Parcoords(
                    line=dict(color=parallel_df[classes[0]],
                              colorscale='Jet',
                              showscale=True,
                              reversescale=True,
                              cmin=0,
                              cmax=size),
                    dimensions=dimension
                )
            ]
            py.iplot(fig)

            # Second plot: bug plotly
            unique_group = df_var[var_categorical_low].unique()
            colors = random_color(len(unique_group))
            data_x = []
            group_label = []
            for i, name in enumerate(unique_group):
                serie = "x_" + str(name)
                serie_x = df_var[df_var[var_categorical_low]
                                 == name][[var_continuous, var_categorical_high]]
                aggregated_serie = serie_x.groupby(
                    var_categorical_high
                ).mean().reset_index().sort_values(by=var_continuous,
                                                   ascending=True)
                data_x.append(aggregated_serie)

            fig1 = make_subplots(rows=1, cols=len(unique_group),
                                 shared_xaxes=True, shared_yaxes=False)

            for i, name in enumerate(unique_group):
                trace1 = {"x": data_x[i][var_continuous],
                          "y": data_x[i][var_categorical_high],
                          "mode": "markers",
                          "marker": dict(
                    color=colors[i],
                ),
                    "name": name,
                    "type": "scatter"
                }
                fig1.append_trace(trace1, 1, i + 1)
            py.iplot(fig1)
            # display(parallel_df)


def list_dropdownHighLow(dic_df):
    """
    """

    l_choice = []
    for key_cont, value_cont in dic_df['var_continuous'].items():

        for key_cat_h, value_cat_h in dic_df['var_categorical_high'].items():

            for key_cat, value_cat in dic_df['var_categorical_low'].items():

                l_choice.append(value_cont['name'] + '-' +
                                value_cat_h['name'] +
                                '-' + value_cat['name'])
    l_choice = ['-'] + l_choice

    return l_choice


def select_catHighLow_eventHandler(df,
                                   dic_df,
                                   cdr=False):
    """
    """

    x_widget = widgets.Dropdown(
        options=list_dropdownHighLow(dic_df=dic_df),
        value='-',
        description='Variable:')

    return interactive(slope_rank,
                       {"manual": True, "auto_display": False},
                       df=fixed(df),
                       dic_multiple=fixed(dic_df),
                       log=False,
                       sample=fixed(False),
                       variables=x_widget,
                       sheetID='',
                       sheetName='',
                       folder='',
                       move_to_drive=False,
                       move_to_drive_batch=False,
                       cdr=fixed(cdr),
                       verbose=fixed(True))

#############################################################################
#############################################################################
#############################################################################
##################### Scatterplot ##############


def computation_scatterplot(df,
                            dic_df,
                            index_Y,
                            index_X,
                            log,
                            sample=False):
    """
    """

    var_y = dic_df['var_Y'][index_Y]['variable_db']
    name_y = dic_df['var_Y'][index_Y]['name']

    drop_y = dic_df['var_Y'][index_Y]['drop_value']

    drop_decil_y = dic_df['var_Y'][index_Y]['drop_decile']

    var_x = dic_df['var_X'][index_X]['variable_db']
    name_x = dic_df['var_X'][index_X]['name']

    drop_x = dic_df['var_X'][index_X]['drop_value']

    drop_decil_x = dic_df['var_X'][index_X]['drop_decile']

    df_var = df[[var_y, var_x]]

    if len(drop_y) > 0:
        # print(drop_continuous)
        df_var = df_var.loc[~df_var.isin(drop_y).any(1)]

    if len(drop_x) > 0:
        # print(drop_continuous)
        df_var = df_var.loc[~df_var.isin(drop_x).any(1)]

    if len(drop_decil_y) > 0:
        # print(drop_decile)
        lower_d = drop_decil_y[0]
        value_lower_d = df_var[var_y].quantile(lower_d)

        higher_d = drop_decil_y[1]
        value_higher_d = df_var[var_y].quantile(higher_d)

        df_var = df_var[df_var[var_y] > value_lower_d]
        df_var = df_var[df_var[var_y] < value_higher_d]

    if len(drop_decil_x) > 0:
        # print(drop_decile)
        lower_d = drop_decil_x[0]
        value_lower_d = df_var[var_x].quantile(lower_d)

        higher_d = drop_decil_x[1]
        value_higher_d = df_var[var_x].quantile(higher_d)

        df_var = df_var[df_var[var_x] > value_lower_d]
        df_var = df_var[df_var[var_x] < value_higher_d]

        # @## Need to append both if both
        df_decile = pd.DataFrame({'Decile': [lower_d, higher_d],
                                  'Value': [value_lower_d, value_higher_d]})

    # return log continuous if log
    if log == 'Y':
        df_1 = df_var.copy()
        try:
            df_1[var_y] = np.log(df_1[var_y])
        except:
            df_1 = np.log(df_1[df_1[var_y] != 0])
    elif log == 'X':
        df_1 = df_var.copy()
        try:
            df_1[var_x] = np.log(df_1[var_x])
        except:
            df_1 = np.log(df_1[df_1[var_x] != 0])
    elif log == 'YX':
        df_1 = df_var.copy()
        try:
            df_1[var_y] = np.log(df_1[var_y])
            df_1[var_x] = np.log(df_1[var_x])
        except:
            df_1 = np.log(df_1[df_1[var_y] != 0])
            df_1 = np.log(df_1[df_1[var_x] != 0])
    else:
        df_1 = df_var.copy()

    dic_int = {
        'output': [
            df_1, var_y, var_x, name_y, name_x
        ]
    }

    return dic_int


def saveToDriveScatter(cdr=False,
                       folder=False,
                       var_x=False,
                       var_y=False,
                       df_scat=False,
                       name_y=False,
                       name_x=False,
                       log=False):

    sns.set_style("white")
    gridobj = sns.lmplot(x=var_x, y=var_y, data=df_scat)
    if log == False:
        name_graph_save = "Scatterplot with line of best fit of " + \
            name_y + ' and ' + name_x
        plt.title(name_graph_save,
                  fontsize=10)
    else:
        name_graph_save = "Scatterplot with line of best fit of " + \
            name_y + ' and ' + name_x + ' in log of ' + log
        plt.title(name_graph_save,
                  fontsize=20)

    name_ = name_graph_save + '.png'
    gridobj.savefig(name_)
    folder_name = folder
    mime_type = "image/png"
    cdr.upload_file_root(mime_type, name_)
    cdr.move_file(file_name=name_,
                  folder_name=folder_name)
    os.remove(name_)


def scatterplot(df,
                dic_multiple,
                variables=False,
                log=False,
                move_to_drive=False,
                move_to_drive_batch=False,
                folder=False,
                cdr=False,
                verbose=False):
    """
    Plot a scatterplot, and save it in plotly
    """

    if move_to_drive_batch:

        max_ = len(dic_multiple['var_Y']) + \
            len(dic_multiple['var_X'])
        # instantiate the bar
        f = IntProgress(min=0, max=max_, description='Loading:')
        display(f)

        for key, value in dic_multiple['var_Y'].items():
            variable_y = value['name']
            f.value += 1
            for key, value in dic_multiple['var_X'].items():
                variable_x = value['name']

                for key, value in dic_multiple['var_Y'].items():
                    if variable_y == value['name']:
                        index_dic_g1 = key
                for key, value in dic_multiple['var_X'].items():
                    if variable_x == value['name']:
                        index_dic_g2 = key

                index_Y = index_dic_g1
                index_X = index_dic_g2

                temp_comp = computation_scatterplot(df=df,
                                                    dic_df=dic_multiple,
                                                    index_Y=index_Y,
                                                    index_X=index_X,
                                                    log=log,
                                                    sample=False)

                df_scat = temp_comp['output'][0]
                var_y = temp_comp['output'][1]
                var_x = temp_comp['output'][2]
                name_y = temp_comp['output'][3]
                name_x = temp_comp['output'][4]

                saveToDriveScatter(cdr=cdr,
                                   folder=folder,
                                   var_x=var_x,
                                   var_y=var_y,
                                   df_scat=df_scat,
                                   name_y=name_y,
                                   name_x=name_x,
                                   log=log)

        f.value += 1

    if verbose:

        regex = r"^[^-]+"
        regex_1 = r"\-(.*)"
        g1 = re.search(regex, variables)[0]
        g2 = re.search(regex_1, variables)[1]
    # dropdown_1_output.clear_output()
        for key, value in dic_multiple['var_Y'].items():
            if g1 == value['name']:
                index_dic_g1 = key
        for key, value in dic_multiple['var_X'].items():
            if g2 == value['name']:
                index_dic_g2 = key

        index_Y = index_dic_g1
        index_X = index_dic_g2

        temp_comp = computation_scatterplot(df=df,
                                            dic_df=dic_multiple,
                                            index_Y=index_Y,
                                            index_X=index_X,
                                            log=log,
                                            sample=False)

        df_scat = temp_comp['output'][0]
        var_y = temp_comp['output'][1]
        var_x = temp_comp['output'][2]
        name_y = temp_comp['output'][3]
        name_x = temp_comp['output'][4]

        if move_to_drive:
            saveToDriveScatter(cdr=cdr,
                               folder=folder,
                               var_x=var_x,
                               var_y=var_y,
                               df_scat=df_scat,
                               name_y=name_y,
                               name_x=name_x,
                               log=log)

        summary_plot = widgets.Output()
        tab_contents = [summary_plot]
        tab = widgets.Tab(tab_contents)
        tab.set_title(0, 'Scatter plot')

        display(tab)

        with summary_plot:

            if log == False:
                name_graph_save = "Scatterplot with line of best fit of " + \
                    name_y + ' and ' + name_x
            else:
                name_graph_save = "Scatterplot with line of best fit of " + \
                    name_y + ' and ' + name_x + ' in log of ' + log

            # iplot(df_scat.iplot(kind='scatter',
            #					asFigure=True,
            #					x=var_x, y=var_y,
            #					mode='markers',
            #					xTitle=name_x, yTitle=name_y,
            #					title=name_graph_save
            #					))

            slope, intercept, r_value, p_value, std_err = stats.linregress(
                df_scat[var_x],
                df_scat[var_y])
            line = slope * df_scat[var_x] + intercept

            fig = go.Figure()
            fig.layout.template = 'plotly_dark'

            fig.add_trace(go.Scattergl(
                x=df_scat[var_x],
                y=df_scat[var_y],
                mode='markers',
                marker=dict(color='rgb(255, 127, 14)',
                            colorscale='Viridis',)
            )
            )

            fig.add_trace(go.Scattergl(
                x=df_scat[var_x],
                y=line,
                mode='lines',
                marker=dict(color='rgb(31, 119, 180)'),
                name='Fit'
            )
            )

            fig['layout']['xaxis'].update(title=name_x)
            fig['layout']['yaxis'].update(title=name_y)

            iplot(fig)


def list_dropdownScatter(dic_df):
    """
    """

    l_choice = []
    for key_cont, value_cont in dic_df['var_Y'].items():

        for key_cat, value_cat in dic_df['var_X'].items():
            l_choice.append(value_cont['name'] + '-' + value_cat['name'])
    l_choice = ['-'] + l_choice

    return l_choice


def select_scatter_eventHandler(df,
                                dic_df,
                                cdr=False
                                ):
    """
    """
    x_widget = widgets.Dropdown(
        options=list_dropdownScatter(dic_df=dic_df),
        value='-',
        description='Variable:')

    return interactive(scatterplot,
                       {"manual": True, "auto_display": False},
                       df=fixed(df),
                       dic_multiple=fixed(dic_df),
                       log=[False, 'Y', 'X', 'YX'],
                       sample=fixed(False),
                       variables=x_widget,
                       folder='',
                       move_to_drive=False,
                       move_to_drive_batch=False,
                       cdr=fixed(cdr),
                       verbose=fixed(True))
#############################################################################
#############################################################################
#############################################################################
##################### Scatter Group 1 ##############


def computation_scatterplotg1(df,
                              dic_df,
                              index_Y,
                              index_X,
                              index_cat,
                              log,
                              aggregationY='sum',
                              aggregationX='sum',
                              sample=False):
    """
    """

    var_y = dic_df['var_Y'][index_Y]['variable_db']
    name_y = dic_df['var_Y'][index_Y]['name']

    drop_y = dic_df['var_Y'][index_Y]['drop_value']
    drop_decil_y = dic_df['var_Y'][index_Y]['drop_decile']

    var_x = dic_df['var_X'][index_X]['variable_db']
    name_x = dic_df['var_X'][index_X]['name']

    var_group = dic_df['var_grouping'][index_cat]['variable_db']
    name_group = dic_df['var_grouping'][index_cat]['name']

    drop_grouping = dic_df['var_grouping'][index_cat]['Drop']

    var_col = dic_df['var_X'][index_X]['color']

    drop_x = dic_df['var_X'][index_X]['drop_value']
    drop_decil_x = dic_df['var_X'][index_X]['drop_decile']

    if len(var_col) > 0:
        df_var = df[[var_y, var_x, var_group, var_col[0]]]
    else:
        df_var = df[[var_y, var_x, var_group]]

    if len(drop_y) > 0:
        # print(drop_continuous)
        df_var = df_var.loc[~df_var.isin(drop_y).any(1)]

    if len(drop_x) > 0:
        # print(drop_continuous)
        df_var = df_var.loc[~df_var.isin(drop_x).any(1)]

    if len(drop_decil_y) > 0:
        # print(drop_decile)
        lower_d = drop_decil_y[0]
        value_lower_d = df_var[var_y].quantile(lower_d)

        higher_d = drop_decil_y[1]
        value_higher_d = df_var[var_y].quantile(higher_d)

        df_var = df_var[df_var[var_y] > value_lower_d]
        df_var = df_var[df_var[var_y] < value_higher_d]

    if len(drop_decil_x) > 0:
        # print(drop_decile)
        lower_d = drop_decil_x[0]
        value_lower_d = df_var[var_x].quantile(lower_d)

        higher_d = drop_decil_x[1]
        value_higher_d = df_var[var_x].quantile(higher_d)

        df_var = df_var[df_var[var_x] > value_lower_d]
        df_var = df_var[df_var[var_x] < value_higher_d]

        # @## Need to append both if both
        df_decile = pd.DataFrame({
            'Decile': [lower_d, higher_d],
            'Value': [value_lower_d, value_higher_d]
        })

    # return log continuous if log
    if log == 'Y':
        df_1 = df_var.copy()
        try:
            df_1[var_y] = np.log(df_1[var_y])
        except:
            df_1 = df_1[df_1[var_y] != 0]
    elif log == 'X':
        df_1 = df_var.copy()
        try:
            df_1[var_x] = np.log(df_1[var_x])
        except:
            df_1 = df_1[df_1[var_x] != 0]
    elif log == 'YX':
        df_1 = df_var.copy()
        try:
            df_1[var_y] = np.log(df_1[var_y])
            df_1[var_x] = np.log(df_1[var_x])
        except:
            df_1 = df_1[df_1[var_y] != 0]
            df_1 = df_1[df_1[var_x] != 0]
    else:
        df_1 = df_var.copy()

    groups = df_1[var_group].unique()
    df_stat = pd.DataFrame({'Y': [], 'X': [], 'group': [], 'n_rows': [],
                            'pearson/R2': [], 'slope': [], 'intercept': [],
                            'p_value': [], 'std_err': []})

    for g in groups:
        df_temp = df_1[df_1[var_group] == g]

        # Compute Stat
        df_stat_temp = regress(X_name=var_x,
                               Y_name=var_y,
                               group=g,
                               x=df_temp[var_x],
                               y=df_temp[var_y])
        df_stat = df_stat.append(df_stat_temp)

    # Aggregate

    df_agg = df_1.groupby(var_group).agg({
        var_y: aggregationY,
        var_x: aggregationX,
    })
    # else:
    #    df_agg =  df_1.groupby([var_group, var_col[0]]).agg({
    #    var_y:aggregationY,
    #    var_x:aggregationX,
    #    })

    dic_int = {
        'output': [
            df_1, df_stat, var_y, var_x, name_y,
            name_x, var_group, name_group, var_col, groups,
            df_agg
        ]
    }

    return dic_int


def saveToDriveScatCat(cdr=False,
                       folder=False,
                       groups=False,
                       var_group=False,
                       df_1=False,
                       var_col=False,
                       var_x=False,
                       var_y=False,
                       name_y=False,
                       name_x=False,
                       name_group=False,
                       log=False,
                       ):

    if len(groups) < 10:

        if len(var_col) > 0:
            g = sns.FacetGrid(df_1, col=var_group,
                              hue=var_col[0])
        else:
            g = sns.FacetGrid(df_1, col=var_group)
        g = (g.map(plt.scatter, var_x, var_y, edgecolor="w")
             .add_legend())

    if log == False:
        name_graph_save = "Scatterplot with line of best fit of " + \
            name_y + ' and ' + name_x + ' grouped by' + name_group
    else:
        name_graph_save = "Scatterplot with line of best fit of " + \
            name_y + ' and ' + name_x + ' grouped by' + name_group +\
            ' in log of ' + log

    name_ = name_graph_save + '.png'
    g.savefig(name_)
    folder_name = folder
    mime_type = "image/png"
    cdr.upload_file_root(mime_type, name_)
    cdr.move_file(file_name=name_,
                  folder_name=folder_name)
    os.remove(name_)


def scatterplot_categorical(df,
                            dic_multiple,
                            variables=False,
                            log=False,
                            aggregationY='sum',
                            aggregationX='sum',
                            folder=False,
                            move_to_drive=False,
                            move_to_drive_batch=False,
                            cdr=False,
                            verbose=False):
    """
    Plot a scatterplot, and save it in plotly
    """

    if move_to_drive_batch:
        max_ = len(dic_multiple['var_Y']) + \
            len(dic_multiple['var_X']) + \
            len(dic_multiple['var_grouping'])
        # instantiate the bar
        f = IntProgress(min=0, max=max_, description='Loading:')
        display(f)
        # dropdown_1_output.clear_output()
        for key, value in dic_multiple['var_Y'].items():
            variable_c = value['name']
            f.value += 1
            for key, value in dic_multiple['var_X'].items():
                variable_cat = value['name']

                for key, value in dic_multiple['var_grouping'].items():
                    variable_low = value['name']

                    for key, value in dic_multiple['var_Y'].items():
                        if variable_c == value['name']:
                            index_dic_g1 = key

                    for key, value in dic_multiple[
                            'var_X'].items():
                        if variable_cat in value['name']:
                            index_dic_g2 = key

                    for key, value in dic_multiple[
                            'var_grouping'].items():
                        if variable_low in value['name']:
                            index_dic_g3 = key

                    index_Y = index_dic_g1
                    index_X = index_dic_g2
                    index_cat = index_dic_g3

                    temp_comp = computation_scatterplotg1(df=df,
                                                          dic_df=dic_multiple,
                                                          index_Y=index_Y,
                                                          index_X=index_X,
                                                          index_cat=index_cat,
                                                          aggregationY=aggregationY,
                                                          aggregationX=aggregationX,
                                                          log=log,
                                                          sample=False)

                    df_1 = temp_comp['output'][0]
                    var_y = temp_comp['output'][2]
                    var_x = temp_comp['output'][3]
                    name_y = temp_comp['output'][4]
                    name_x = temp_comp['output'][5]
                    var_group = temp_comp['output'][6]
                    name_group = temp_comp['output'][7]
                    var_col = temp_comp['output'][8]
                    groups = temp_comp['output'][9]

                    # Make graphs
                    saveToDriveScatCat(cdr=cdr,
                                       folder=folder,
                                       groups=groups,
                                       var_group=var_group,
                                       df_1=df_1,
                                       var_col=var_col,
                                       var_x=var_x,
                                       var_y=var_y,
                                       name_y=name_y,
                                       name_x=name_x,
                                       name_group=name_group,
                                       log=log
                                       )

                    f.value += 1
        f.value += 1
    if verbose:

        regex = r"^[^-]+"
        regex_1 = r"\-(.*?)\-"
        regex_2 = r"\-(.*)"
        # Very ugly way
        g1 = re.search(regex, variables)[0]
        g2 = re.search(regex_1, variables)[1]
        g3 = re.search(regex_2, variables)[1]
        g4 = re.search(regex_2, g3)[1]

        for key, value in dic_multiple['var_Y'].items():
            if g1 == value['name']:
                index_dic_g1 = key

        for key, value in dic_multiple['var_X'].items():
            if g2 == value['name']:
                index_dic_g2 = key

        for key, value in dic_multiple['var_grouping'].items():
            if g4 == value['name']:
                index_dic_g3 = key

        index_Y = index_dic_g1
        index_X = index_dic_g2
        index_cat = index_dic_g3

        temp_comp = computation_scatterplotg1(df=df,
                                              dic_df=dic_multiple,
                                              index_Y=index_Y,
                                              index_X=index_X,
                                              index_cat=index_cat,
                                              aggregationY=aggregationY,
                                              aggregationX=aggregationX,
                                              log=log,
                                              sample=False)

        df_1 = temp_comp['output'][0]
        df_stat = temp_comp['output'][1]
        var_y = temp_comp['output'][2]
        var_x = temp_comp['output'][3]
        name_y = temp_comp['output'][4]
        name_x = temp_comp['output'][5]
        var_group = temp_comp['output'][6]
        name_group = temp_comp['output'][7]
        var_col = temp_comp['output'][8]
        groups = temp_comp['output'][9]
        df_agg = temp_comp['output'][10]

        if move_to_drive:
            saveToDriveScatCat(cdr=cdr,
                               folder=folder,
                               groups=groups,
                               var_group=var_group,
                               df_1=df_1,
                               var_col=var_col,
                               var_x=var_x,
                               var_y=var_y,
                               name_y=name_y,
                               name_x=name_x,
                               name_group=name_group,
                               log=log
                               )

        summary_ = widgets.Output()
        summary_plot = widgets.Output()
        #summary_plot_color = widgets.Output()
        summary_agg = widgets.Output()

        tab_contents = [summary_agg, summary_plot, summary_]
        tab = widgets.Tab(tab_contents)
        tab.set_title(0, 'Plot scatter, aggregated')
        tab.set_title(1, 'Plot scatter')
        tab.set_title(2, 'Linear regression')

        display(tab)

        with summary_:
            unique_color = df_1[var_group].unique()
            unique_group_c = setcolors(unique_color)
            df_stat = df_stat.reset_index()

            cm = sns.light_palette("green", as_cmap=True)
            temp = (df_stat.style
                    .bar(subset=['pearson/R2', 'n_rows'], align='mid',
                         color=['#d65f5f', '#5fba7d'])
                    .format({
                        'std_err': '±{:.2f}'})
                    .applymap(applycolors,
                              l_colors=unique_group_c,
                              subset=['group'])
                    )
            display(temp)
        with summary_plot:
            if len(groups) < 10:

                if len(var_col) > 0:
                    g = sns.FacetGrid(df_1, col=var_group,
                                      hue=var_col[0])
                else:
                    g = sns.FacetGrid(df_1, col=var_group)
                g = (g.map(plt.scatter, var_x, var_y, edgecolor="w")
                     .add_legend())
            plt.show()

        with summary_agg:
            df_agg = df_agg.reset_index()

            data = [
                go.Scatter(
                    x=df_agg[var_x],
                    y=df_agg[var_y],
                    mode='markers+text',
                    text=df_agg[var_group],
                    textposition='bottom center'
                )
            ]

            if log == False:
                name_graph_save = "Scatterplot of " + \
                    aggregationY + '-' + name_y + ' and ' + aggregationX + '-' +\
                    name_x + ' grouped by ' + name_group
            else:
                name_graph_save = "Scatterplot " + \
                    aggregationY + '-' + name_y + ' and ' + aggregationX + '-' +\
                    name_x + ' grouped by ' + name_group +\
                    ' in log of ' + log

            layout = go.Layout(
                title=name_graph_save
            )

            fig = go.Figure(data=data, layout=layout)
            fig.layout.template = 'plotly_dark'
            iplot(fig)


def list_dropdownscatterG(dic_df):
    """
    """

    l_choice = []
    for key_cont, value_cont in dic_df['var_Y'].items():

        for key_cat_h, value_cat_h in dic_df['var_X'].items():

            for key_cat, value_cat in dic_df['var_grouping'].items():

                l_choice.append(value_cont['name'] + '-' +
                                value_cat_h['name'] +
                                '-' + value_cat['name'])
    l_choice = ['-'] + l_choice

    return l_choice


def select_scatterGroup_eventHandler(df,
                                     dic_df,
                                     cdr=False
                                     ):
    """
    """

    x_widget = widgets.Dropdown(
        options=list_dropdownscatterG(dic_df=dic_df),
        value='-',
        description='Variable:')

    return interactive(scatterplot_categorical,
                       {"manual": True, "auto_display": False},
                       df=fixed(df),
                       dic_multiple=fixed(dic_df),
                       log=[False, 'Y', 'X', 'YX'],
                       sample=fixed(False),
                       variables=x_widget,
                       aggregationY=['sum', 'median', 'mean', 'min', 'max'],
                       aggregationX=['sum', 'median', 'mean', 'min', 'max'],
                       folder='',
                       move_to_drive=False,
                       move_to_drive_batch=False,
                       cdr=fixed(cdr),
                       verbose=fixed(True))

#############################################################################
#############################################################################
#############################################################################
##################### Scatterplot G1G2 ##############


def computation_scatterplotg1g2(df,
                                dic_df,
                                index_Y,
                                index_X,
                                index_cat,
                                var_col,
                                log,
                                sample=False):
    """
    """

    var_y = dic_df['var_Y'][index_Y]['variable_db']
    name_y = dic_df['var_Y'][index_Y]['name']

    drop_y = dic_df['var_Y'][index_Y]['drop_value']
    drop_decil_y = dic_df['var_Y'][index_Y]['drop_decile']

    var_x = dic_df['var_X'][index_X]['variable_db']
    name_x = dic_df['var_X'][index_X]['name']

    var_group = dic_df['var_grouping'][index_cat]['variable_db']
    name_group = dic_df['var_grouping'][index_cat]['name']

    drop_grouping = dic_df['var_grouping'][index_cat]['Drop']

    #var_col = dic_df['var_X'][index_X]['color']

    drop_x = dic_df['var_X'][index_X]['drop_value']
    drop_decil_x = dic_df['var_X'][index_X]['drop_decile']

    if var_col != False:
        df_var = df[[var_y, var_x, var_group, var_col]]
    else:
        df_var = df[[var_y, var_x, var_group]]

    if len(drop_y) > 0:
        # print(drop_continuous)
        df_var = df_var.loc[~df_var.isin(drop_y).any(1)]

    if len(drop_x) > 0:
        # print(drop_continuous)
        df_var = df_var.loc[~df_var.isin(drop_x).any(1)]

    if len(drop_decil_y) > 0:
        # print(drop_decile)
        lower_d = drop_decil_y[0]
        value_lower_d = df_var[var_y].quantile(lower_d)

        higher_d = drop_decil_y[1]
        value_higher_d = df_var[var_y].quantile(higher_d)

        df_var = df_var[df_var[var_y] > value_lower_d]
        df_var = df_var[df_var[var_y] < value_higher_d]

    if len(drop_decil_x) > 0:
        # print(drop_decile)
        lower_d = drop_decil_x[0]
        value_lower_d = df_var[var_x].quantile(lower_d)

        higher_d = drop_decil_x[1]
        value_higher_d = df_var[var_x].quantile(higher_d)

        df_var = df_var[df_var[var_x] > value_lower_d]
        df_var = df_var[df_var[var_x] < value_higher_d]

        # @## Need to append both if both
        df_decile = pd.DataFrame({
            'Decile': [lower_d, higher_d],
            'Value': [value_lower_d, value_higher_d]
        })

    # return log continuous if log
    if log == 'Y':
        df_1 = df_var.copy()
        try:
            df_1[var_y] = np.log(df_1[var_y])
        except:
            df_1 = df_1[df_1[var_y] != 0]
    elif log == 'X':
        df_1 = df_var.copy()
        try:
            df_1[var_x] = np.log(df_1[var_x])
        except:
            df_1 = df_1[df_1[var_x] != 0]
    elif log == 'YX':
        df_1 = df_var.copy()
        try:
            df_1[var_y] = np.log(df_1[var_y])
            df_1[var_x] = np.log(df_1[var_x])
        except:
            df_1 = df_1[df_1[var_y] != 0]
            df_1 = df_1[df_1[var_x] != 0]
    else:
        df_1 = df_var.copy()

    if var_col != False:
        df_dot = df_1.groupby([var_group, var_col]).agg({
            var_y: ['mean', 'median', 'sum'],
            var_x: ['mean', 'median', 'sum']})
    else:
        df_dot = df_1.groupby([var_group]).agg({
            var_y: ['mean', 'median', 'sum'],
            var_x: ['mean', 'median', 'sum']})

    df_dot.columns = df_dot.columns.droplevel()
    df_dot.columns = [str(var_y) + "_mean",
                      str(var_y) + "_median",
                      str(var_y) + "_sum",
                      str(var_x) + "_mean",
                      str(var_x) + "_median",
                      str(var_x) + "_sum"
                      ]

    df_dot = df_dot.fillna('')
    df_dot = df_dot.reset_index()

    dic_int = {
        'output': [
            df_1, df_dot, var_y, var_x, name_y,
            name_x, var_group, name_group, var_col
        ]
    }

    return dic_int


def saveToDriveScatterG1G2(cdr=False,
                           sheetID=False,
                           sheetName=False,
                           df_dot=False,
                           LatestRow=False):
    """
    """
    len_btw = 5
    nb_cols = df_dot.shape[1]
    n_rows = df_dot.shape[0]
    begin = LatestRow + 4
    end = begin + n_rows + 1

    for i, letter in enumerate(alphabet):
        if i == nb_cols:
            range_1_letter = letter

    range_dot = "A" + str(LatestRow + 4) + \
        ":" + str(range_1_letter) + str(end)

    table_output = {
        'df_dot': df_dot.to_numpy().tolist(),
        'range_dot': range_dot,
        'header_dot': list(df_dot),
    }

    cdr.add_data_to_spreadsheet(data=table_output['df_dot'],
                                sheetID=sheetID,
                                sheetName=sheetName,
                                rangeData=table_output['range_dot'],
                                headers=table_output['header_dot'])


def scatter_g1_g2(df,
                  dic_multiple,
                  log=False,
                  variables=False,
                  var_col=False,
                  sheetID=False,
                  sheetName=False,
                  folder=False,
                  move_to_drive=False,
                  move_to_drive_batch=False,
                  cdr=False,
                  verbose=False):
    """
    Filter has to be a higher level than the group

    """

    # Extract each component of the dictionary

    if move_to_drive_batch:
        max_ = len(dic_multiple['var_Y']) + \
            len(dic_multiple['var_X']) + \
            len(dic_multiple['var_grouping'])
        # instantiate the bar
        f = IntProgress(min=0, max=max_, description='Loading:')
        display(f)
        # dropdown_1_output.clear_output()
        for key, value in dic_multiple['var_Y'].items():
            variable_c = value['name']
            f.value += 1
            for key, value in dic_multiple['var_X'].items():
                variable_cat = value['name']

                for key, value in dic_multiple['var_grouping'].items():
                    variable_low = value['name']

                    for key, value in dic_multiple['var_Y'].items():
                        if variable_c == value['name']:
                            index_dic_g1 = key

                    for key, value in dic_multiple[
                            'var_X'].items():
                        if variable_cat == value['name']:
                            index_dic_g2 = key

                    for key, value in dic_multiple[
                            'var_grouping'].items():
                        if variable_low == value['name']:
                            index_dic_g3 = key

                    index_Y = index_dic_g1
                    index_X = index_dic_g2
                    index_cat = index_dic_g3

                    temp_comp = computation_scatterplotg1g2(df=df,
                                                            dic_df=dic_multiple,
                                                            index_Y=index_Y,
                                                            index_X=index_X,
                                                            index_cat=index_cat,
                                                            var_col=var_col,
                                                            log=log,
                                                            sample=False)

                    df_dot = temp_comp['output'][1]
                    var_y = temp_comp['output'][2]
                    var_x = temp_comp['output'][3]
                    name_y = temp_comp['output'][4]
                    name_x = temp_comp['output'][5]
                    var_group = temp_comp['output'][6]
                    name_group = temp_comp['output'][7]
                    var_col = temp_comp['output'][8]

                    LatestRow = cdr.getLatestRow(
                        sheetID=sheetID, sheetName=sheetName)

                    saveToDriveScatterG1G2(cdr=cdr,
                                           sheetID=sheetID,
                                           sheetName=sheetName,
                                           df_dot=df_dot,
                                           LatestRow=LatestRow)

                f.value += 1
            f.value += 1
    if verbose:
        regex = r"^[^-]+"
        regex_1 = r"\-(.*?)\-"
        regex_2 = r"\-(.*)"
        # Very ugly way
        g1 = re.search(regex, variables)[0]
        g2 = re.search(regex_1, variables)[1]
        g3 = re.search(regex_2, variables)[1]
        g4 = re.search(regex_2, g3)[1]

        for key, value in dic_multiple['var_Y'].items():
            if g1 == value['name']:
                index_dic_g1 = key

        for key, value in dic_multiple['var_X'].items():
            if g2 == value['name']:
                index_dic_g2 = key

        for key, value in dic_multiple['var_grouping'].items():
            if g4 == value['name']:
                index_dic_g3 = key

        index_Y = index_dic_g1
        index_X = index_dic_g2
        index_cat = index_dic_g3

        temp_comp = computation_scatterplotg1g2(df=df,
                                                dic_df=dic_multiple,
                                                index_Y=index_Y,
                                                index_X=index_X,
                                                index_cat=index_cat,
                                                var_col=var_col,
                                                log=log,
                                                sample=False)

        df_1 = temp_comp['output'][0]
        df_dot = temp_comp['output'][1]
        var_y = temp_comp['output'][2]
        var_x = temp_comp['output'][3]
        name_y = temp_comp['output'][4]
        name_x = temp_comp['output'][5]
        var_group = temp_comp['output'][6]
        name_group = temp_comp['output'][7]
        var_col = temp_comp['output'][8]

        if move_to_drive:
            LatestRow = cdr.getLatestRow(
                sheetID=sheetID, sheetName=sheetName)

            saveToDriveScatterG1G2(cdr=cdr,
                                   sheetID=sheetID,
                                   sheetName=sheetName,
                                   df_dot=df_dot,
                                   LatestRow=LatestRow)

        summary_plot = widgets.Output()
        summary_plot1 = widgets.Output()
        summary_plot2 = widgets.Output()

        tab_contents = [summary_plot2, summary_plot, summary_plot1]
        tab = widgets.Tab(tab_contents)

        tab.set_title(0, 'Scatter plot, filter')
        tab.set_title(1, 'Scatter plot')
        tab.set_title(2, 'Scatter plot, color')

        display(tab)

        for i, stat in enumerate(['_mean', '_median', '_sum']):
            filter_col = [col for col in df_dot if col.endswith(stat)]
            filter_col.append(var_col)
            filter_col.append(var_group)
            df_scat = df_dot[filter_col]

            y = var_y + stat
            x = var_x + stat

            if log == False:
                name_graph_save = "Scatterplot with line of best fit of " + \
                    name_y + ' and ' + name_x + ' aggregated by' + stat + \
                    ' for ' + var_group + ' and ' + var_col
            else:
                name_graph_save = "Scatterplot with line of best fit of " + \
                    name_y + ' and ' + name_x + ' aggregated by' + stat +\
                    ' for ' + var_group + ' and ' + var_col + ' in log of ' + log

            with summary_plot:
                # if stat == '_sum':
                ax = sns.regplot(x=x, y=y, data=df_scat)
                ax.set_title(name_graph_save)
                plt.show()

                if move_to_drive:
                    name_ = name_graph_save + '.png'
                    ax.get_figure().savefig(name_)
                    folder_name = folder
                    mime_type = "image/png"
                    cdr.upload_file_root(mime_type, name_)
                    cdr.move_file(file_name=name_,
                                  folder_name=folder_name)
                    os.remove(name_)

            with summary_plot1:
                # if stat == '_sum':
                if len(df_scat[var_col].unique()) > 10:
                    legend = False
                else:
                    legend = 'full'
                ax = sns.scatterplot(x=x, y=y,
                                     hue=var_col,
                                     data=df_scat,
                                     legend=legend)
                ax.set_title(name_graph_save)
                plt.show()

                if move_to_drive:
                    name_ = name_graph_save + '.png'
                    ax.get_figure().savefig(name_)
                    folder_name = folder
                    mime_type = "image/png"
                    cdr.upload_file_root(mime_type, name_)
                    cdr.move_file(file_name=name_,
                                  folder_name=folder_name)
                    os.remove(name_)

            if stat == '_sum':
                with summary_plot2:
                    # df_scat_r = df_scat.pivot(index=var_col, columns=var_group,
                                 # values= [x, y])
                    df_scat_f = df_scat

                    iplot({
                        'data': [
                            {
                                'x': df_scat_f[df_scat_f[var_col] == g][x],
                                'y': df_scat_f[df_scat_f[var_col] == g][y],
                                'name': g, 'mode': 'markers',
                                'text':df_scat_f[var_group],
                            } for g in list(df_scat_f[var_col].unique())
                        ],
                        'layout': {
                            'xaxis': {'title': name_x},
                            'yaxis': {'title': name_y}
                        }
                    })

                    # display(df_scat)


def select_scatterGroup2_eventHandler(df,
                                      dic_df,
                                      cdr=False
                                      ):
    """
    """
    l_filter = list(df.select_dtypes(include='object'))
    l_filter = [False] + l_filter
    x_widget = widgets.Dropdown(
        options=list_dropdownscatterG(dic_df=dic_df),
        value='-',
        description='Variable:')

    return interactive(scatter_g1_g2,
                       {"manual": True, "auto_display": False},
                       df=fixed(df),
                       dic_multiple=fixed(dic_df),
                       log=[False, 'Y', 'X', 'YX'],
                       sample=fixed(False),
                       variables=x_widget,
                       var_col=l_filter,
                       sheetID='',
                       sheetName='',
                       folder='',
                       move_to_drive=False,
                       move_to_drive_batch=False,
                       cdr=fixed(cdr),
                       verbose=fixed(True))

#############################################################################
#############################################################################
#############################################################################
##################### Categorical ##############


def prepare_df_choice(df,
                      dic_multiple,
                      index=['V0', 'V0'],
                      simple=True):
    """
    Prepare the dataset for the Chi square function and CA function
    index is the index in the dic_multiple to return the correct
    var_columns

    for instance, if index is V3, then the function will choose the third index
    By default V0. It allows to compute in batch behaviour variables
    """

    # Need to know wich index: row or colums
    # if len(dic_multiple['var_rows']) > 1:
    index_r = index[1]
    index_c = index[0]

    var_columns = dic_multiple['var_columns'][index_c]['variable_db']
    var_rows = dic_multiple['var_rows'][index_r]['values']
    new_name_column = dic_multiple['var_columns'][index_c]['name']
    new_name_rows = dic_multiple['var_rows'][index_r]['variable_db']
    rows_to_drop = dic_multiple['var_rows'][index_r]['Drop']
    # else:
    #    index_r = 'V0'
    #    var_columns = dic_multiple['var_columns'][index]['variable_db']
    #    var_rows = dic_multiple['var_rows'][index_r]['values']
    #    new_name_column = dic_multiple['var_columns'][index]['name']
    #    new_name_rows = dic_multiple['var_rows'][index_r]['variable_db']
    #    rows_to_drop = dic_multiple['var_rows'][index_r]['Drop']

    # Select columns
    col_needed = []
    if simple == True:
        col_needed.append(var_columns)
        col_needed.append(new_name_rows)
    else:
        for n in var_rows:
            col_needed.append(n)
        col_needed.append(var_columns)
    df_question = df[col_needed]

    # Drop if needed
    #new_name = dic_multiple['var_columns']['name']
    try:
        drop_var = dic_multiple['var_columns'][index]['Drop']
    except:
        drop_var = dic_multiple['var_columns'][index_c]['Drop']
    try:
        for to_drop in drop_var:
            df_question = df_question[df_question[var_columns] != to_drop]
    except:
        pass

        # Drop rows
    if len(rows_to_drop) > 0:
        print(rows_to_drop)
        df_question = df_question.loc[~df_question.isin(rows_to_drop).any(1)]

    if simple == True:
        df_question = pd.crosstab(df_question[new_name_rows],
                                  df_question[var_columns])

    else:
        df_question = df_question.apply(pd.to_numeric, errors='ignore')
        df_question = df_question.set_index(var_columns).stack().reset_index(
        ).sort_values(by=[var_columns,
                          'level_1'])
        df_question.columns = [new_name_column, new_name_rows, 'answer']
        df_question = df_question.groupby([new_name_rows, new_name_column]).agg({
            'answer': 'sum'}).rename(
            columns={'answer': 'sum'}).unstack()

        df_question.columns = df_question.columns.droplevel()

    df_question = df_question.reset_index()
    df_question.columns = df_question.columns.tolist()
    df_question = df_question.set_index(new_name_rows)

    return df_question

#############################################################################
#############################################################################
#############################################################################
##################### Categorical ##############


def categorical_analysis(df,
                         name_column,
                         ca=False):
    """
    Compute Chi test a Correspondence Analysis

    """

    # Total
    n_rows = df.shape[0]
    # Add total Count

    slide = df.reset_index()
    df_total = df.sum(axis=1).reset_index().iloc[:, 1:]
    headers_data = slide.columns
    index_name = headers_data[0]
    slide_chi = slide.iloc[:, 1:slide.shape[1] + 1]
    slide_perc_rows = pd.concat([slide[index_name],
                                 slide_chi.apply(lambda r: r / r.sum(),
                                                 axis=1)], axis=1)

    slide_perc_col = pd.concat([slide[index_name],
                                slide_chi.apply(lambda r: r / r.sum(),
                                                axis=0)], axis=1)

    # contingency table
    stat, p, dof, expected = chi2_contingency(slide_chi)
    # interpret test-statistic
    prob = 0.95
    critical = chi2.ppf(prob, dof)
    # interpret p-value
    alpha = round(1.0 - prob, 2)
    sign = 'significance=%.3f, p=%.3f'
    if p <= alpha:
        result = 'Dependent (reject H0)'
    else:
        result = 'Independent (fail to reject H0)'
    l_resut = [[dof, alpha, result, np.array(df.values.sum()).tolist()]]

    # Compute post hoc test
    pearson_resid = (slide_chi - expected) / np.sqrt(expected)

    contribution = np.array(np.around(pearson_resid ** 2 / stat, 2))

    ###
    pearson_resid_df = pd.DataFrame(pearson_resid, columns=headers_data[1:])
    pearson_resid_df = pearson_resid_df.set_index(slide[index_name])
    pearson_resid_df = pearson_resid_df.reset_index()

    contribution_df = pd.DataFrame(contribution, columns=headers_data[1:])
    contribution_df = contribution_df.set_index(slide[index_name])
    contribution_df = contribution_df.reset_index()

    # get range for Google Sheet
    # alphabet = ['A', 'B', 'C', 'D','E','F','G','H','I','J','K','L','M','N','O',
    #            'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
    #            'Z', 'AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH','AI']

    len_btw = 5
    nb_cols = df.shape[1] + 2
    nb_rows = n_rows + 1

    dic_int = {
        'output': [
            slide, df_total, slide_perc_rows, slide_perc_col,
            pearson_resid_df, contribution_df, l_resut, headers_data,
            name_column
        ]
    }

    return dic_int


def saveToDriveCategorical(cdr=False,
                           sheetID=False,
                           sheetName=False,
                           folder=False,
                           slide=False,
                           df_total=False,
                           slide_perc_rows=False,
                           slide_perc_col=False,
                           pearson_resid_df=False,
                           contribution_df=False,
                           l_resut=False,
                           LatestRow=False,
                           column_name=False,
                           headers_data=False):
    """
    """
    df_count = slide.shape[0]
    len_btw = 5
    nb_cols = slide.shape[1] + 2
    nb_rows = df_count + 1
    begin = LatestRow + 4
    n_end_row = begin + 3 * len_btw + nb_rows * 3

    for i, letter in enumerate(alphabet):
        if i == nb_cols + 1:
            range_2_letter = letter
        if i + 1 == nb_cols:
            range_letter = letter
        if i + 2 == nb_cols:
            range_1_letter = letter

    dic_range = {
        'range_raw':
            "A" + str(begin) + ':' + str(range_1_letter) +
            str(begin + nb_rows),
            'total_raw':
            str(range_letter) + str(begin) + ':' + str(range_letter) +
            str(begin + nb_rows),
            'range_prop_raws':
            "A" + str(begin + 1 * (nb_rows + len_btw)) + ":" +
            range_1_letter + str(begin + 1 * len_btw + nb_rows * 2),
            'range_prop_col':
            str(range_2_letter) + str(begin + 1 * (nb_rows + len_btw))
            + ":CZ" + str(begin + 1 * len_btw + nb_rows * 2),
            'range_pearson':
            "A" + str(begin + 2 * (nb_rows + len_btw)) + ":" +
            range_1_letter + str(begin + 2 * len_btw + nb_rows * 3),
            'range_contribution':
            str(range_2_letter) + str(begin + 2 * (nb_rows + len_btw))
            + ":CZ" + str(begin + 2 * len_btw + nb_rows * 3),
            'range_test':
            range_2_letter + str(begin) + ":CZ" + str(begin + 1),
            'range_name_column':
            range_2_letter + str(begin + 3) + ":CZ" + str(begin + 4)
    }

    table_output = {
        'table_count': slide,
        'table_total': df_total,
        'table_proportion_rows':
            slide_perc_rows,
            'table_proportion_col': slide_perc_col,
            'pearson_resid': pearson_resid_df,
            'contribution': contribution_df,
            'result_test': l_resut,
            'ranges': dic_range,
            'header_data': list(headers_data),
            'header_chi': ['dof', 'p_value', 'result', 'nb rows'],
            'name_column': [column_name]
    }

    cdr.add_data_to_spreadsheet(
        data=table_output['table_count'],
        sheetID=sheetID,
        sheetName=sheetName,
		detectRange = False,
        rangeData=table_output['ranges']['range_raw']
        #headers=table_output['header_data']
		)

    # total count
    cdr.add_data_to_spreadsheet(
        data=table_output['table_total'],
        sheetID=sheetID,
        sheetName=sheetName,
		detectRange = False,
        rangeData=table_output['ranges']['total_raw']
        #headers=['Total']
		)

    # table_proportion_rows
    cdr.add_data_to_spreadsheet(
        data=table_output['table_proportion_rows'],
        sheetID=sheetID,
        sheetName=sheetName,
		detectRange = False,
        rangeData=table_output['ranges']['range_prop_raws']
        #headers=table_output['header_data']
		)

    # table_proportion_col
    cdr.add_data_to_spreadsheet(
        data=table_output['table_proportion_col'],
        sheetID=sheetID,
        sheetName=sheetName,
		detectRange = False,
        rangeData=table_output['ranges']['range_prop_col']
        #headers=table_output['header_data']
		)

    # pearson_resid
    cdr.add_data_to_spreadsheet(
        data=table_output['pearson_resid'],
        sheetID=sheetID,
        sheetName=sheetName,
		detectRange = False,
        rangeData=table_output['ranges']['range_pearson']
        #headers=table_output['header_data']
		)

    # contribution
    cdr.add_data_to_spreadsheet(
        data=table_output['contribution'],
        sheetID=sheetID,
        sheetName=sheetName,
		detectRange = False,
        rangeData=table_output['ranges']['range_contribution']
        #headers=table_output['header_data']
		)

    # result_test
    cdr.add_data_to_spreadsheet(
        data= pd.DataFrame(table_output['result_test']),
        sheetID=sheetID,
        sheetName=sheetName,
		detectRange = False,
        rangeData=table_output['ranges']['range_test']
        #headers=table_output['header_chi']
		)

    # Name column
    cdr.add_data_to_spreadsheet(
        data=[table_output['name_column']],
        sheetID=sheetID,
        sheetName=sheetName,
		detectRange = False,
        rangeData=table_output['ranges']['range_name_column']
        #headers=['name_column']
		)


def categorical(df,
                dic_multiple,
                variables=False,
                ca=False,
                sheetID=False,
                sheetName=False,
                folder=False,
                move_to_drive=False,
                move_to_drive_batch=False,
                cdr=False,
                verbose=False):
    """
    """

    if move_to_drive_batch:

        max_ = len(dic_multiple['var_columns']) + \
            len(dic_multiple['var_rows'])
        # instantiate the bar
        f = IntProgress(min=0, max=max_, description='Loading:')
        display(f)

        for key, value in dic_multiple['var_columns'].items():
            variable_y = value['name']
            column_name = value['name']
            f.value += 1
            for key, value in dic_multiple['var_rows'].items():
                variable_x = value['name']

                for key, value in dic_multiple['var_columns'].items():
                    if variable_y == value['name']:
                        index_dic_g1 = key
                for key, value in dic_multiple['var_rows'].items():
                    if variable_x == value['name']:
                        index_dic_g2 = key

                index_Y = index_dic_g1
                index_X = index_dic_g2

                # Prepare dataset
                df_count = prepare_df_choice(
                    df=df,
                    dic_multiple=dic_multiple,
                    index=[index_Y, index_X],
                    simple=True)

                cat_analysis = categorical_analysis(
                    df=df_count, name_column=[index_Y, index_X], ca=False)

                slide = cat_analysis['output'][0]
                df_total = cat_analysis['output'][1]
                slide_perc_rows = cat_analysis['output'][2]
                slide_perc_col = cat_analysis['output'][3]
                pearson_resid_df = cat_analysis['output'][4]
                contribution_df = cat_analysis['output'][5]
                l_resut = cat_analysis['output'][6]
                headers_data = cat_analysis['output'][7]
                name_column = cat_analysis['output'][8]

                # Move to drive
                LatestRow = cdr.getLatestRow(
                    sheetID=sheetID, sheetName=sheetName)

                saveToDriveCategorical(cdr=cdr,
                                       sheetID=sheetID,
                                       sheetName=sheetName,
                                       folder=folder,
                                       slide=slide,
                                       df_total=df_total,
                                       slide_perc_rows=slide_perc_rows,
                                       slide_perc_col=slide_perc_col,
                                       pearson_resid_df=pearson_resid_df,
                                       contribution_df=contribution_df,
                                       l_resut=l_resut,
                                       LatestRow=LatestRow,
                                       column_name=column_name,
                                       headers_data=headers_data)

        f.value += 1
    if verbose:
        regex = r"^[^-]+"
        regex_1 = r"\-(.*)"
        g1 = re.search(regex, variables)[0]
        g2 = re.search(regex_1, variables)[1]

        for key, value in dic_multiple['var_columns'].items():
            if g1 == value['name']:
                index_dic_g1 = key
                column_name = value['name']
        for key, value in dic_multiple['var_rows'].items():
            if g2 == value['name']:
                index_dic_g2 = key

        df_count = prepare_df_choice(
            df=df,
            dic_multiple=dic_multiple,
            index=[index_dic_g1, index_dic_g2],
            simple=True)

        cat_analysis = categorical_analysis(
            df=df_count, name_column=[index_dic_g1, index_dic_g2], ca=False)

        slide = cat_analysis['output'][0]
        df_total = cat_analysis['output'][1]
        slide_perc_rows = cat_analysis['output'][2]
        slide_perc_col = cat_analysis['output'][3]
        pearson_resid_df = cat_analysis['output'][4]
        contribution_df = cat_analysis['output'][5]
        l_resut = cat_analysis['output'][6]
        headers_data = cat_analysis['output'][7]
        name_column = cat_analysis['output'][8]

        if move_to_drive:
            LatestRow = cdr.getLatestRow(
                sheetID=sheetID, sheetName=sheetName)

            saveToDriveCategorical(cdr=cdr,
                                   sheetID=sheetID,
                                   sheetName=sheetName,
                                   folder=folder,
                                   slide=slide,
                                   df_total=df_total,
                                   slide_perc_rows=slide_perc_rows,
                                   slide_perc_col=slide_perc_col,
                                   pearson_resid_df=pearson_resid_df,
                                   contribution_df=contribution_df,
                                   l_resut=l_resut,
                                   LatestRow=LatestRow,
                                   column_name=column_name,
                                   headers_data=headers_data)

        summary_raw = widgets.Output()
        #summary_count = widgets.Output()
        summary_proportion = widgets.Output()
        summary_proportion_col = widgets.Output()
        summary_pearson = widgets.Output()
        summary_contribution = widgets.Output()
        summary_result = widgets.Output()

        with summary_raw:
            df_total.columns = ['total']
            total = pd.concat([slide, df_total], axis=1).sort_values(
                by='total', ascending=False)

            # Sort all rows according to this sorting
            custom_index = total.index

            total = total.style.bar(subset=['total'], color='#d65f5f')

            display(total)

        # with summary_count:
        #    display(df_total)

        with summary_proportion:

            slide_perc_rows = slide_perc_rows.reindex(custom_index)

            cm = sns.light_palette("#00bfff", as_cmap=True)
            slide_perc_rows = slide_perc_rows.style.background_gradient(
                cmap=cm)
            display(slide_perc_rows)

        with summary_proportion_col:

            cm = sns.light_palette("#00bfff", as_cmap=True)
            slide_perc_col = slide_perc_col.reindex(custom_index)
            slide_perc_col = slide_perc_col.style.background_gradient(cmap=cm)

            display(slide_perc_col)

        with summary_pearson:
            pearson_resid_df = pearson_resid_df.reindex(custom_index)

            cm = sns.light_palette("#00bfff", as_cmap=True)
            # pearson_resid_df = pearson_resid_df.style.bar(align='mid', color=['#d65f5f', '#5fba7d'])
            pearson_resid_df = pearson_resid_df.style.background_gradient(
                cmap=cm)
            display(pearson_resid_df)

        with summary_contribution:
            contribution_df = contribution_df.reindex(custom_index)
            display(contribution_df)

        with summary_result:
            display(l_resut)

        if ca:

            ca = ca_compute.compute_ca(df_count)
            ca_computed = ca.correspondance_analysis()

            fig_2 = ca_compute.row_focus_coordinates(
                df_x=ca_computed['pc_rows'],
                df_y=ca_computed['pc_columns'],
                variance_explained=ca_computed['variance_explained'],
                export_data=True)

            summary_ca = widgets.Output()
            tab = widgets.Tab([summary_raw, summary_proportion,
                               summary_proportion_col, summary_pearson,
                               summary_contribution, summary_result, summary_ca])
            tab.set_title(6, 'Correspondence Anasylis')
            with summary_ca:
                display(fig_2['figure'])
                if move_to_drive:
                    name_ = 'Correspondence Anasylis' + '.png'
                    fig_2['figure'].savefig(name_)
                    folder_name = folder
                    mime_type = "image/png"
                    cdr.upload_file_root(mime_type, name_)
                    cdr.move_file(file_name=name_,
                                  folder_name=folder_name)
                    os.remove(name_)

        else:
            tab = widgets.Tab([summary_raw, summary_proportion,
                               summary_proportion_col, summary_pearson,
                               summary_contribution, summary_result])

        tab.set_title(0, 'table_count')
        #tab.set_title(1, 'table_total')
        tab.set_title(1, 'table_proportion_rows')
        tab.set_title(2, 'table_proportion_col')
        tab.set_title(3, 'pearson_resid')
        tab.set_title(4, 'contribution')
        tab.set_title(5, 'result_test')

        display(tab)


def list_dropdownCat(dic_df):

    l_choice = []
    for key_cont, value_cont in dic_df['var_columns'].items():

        for key_cat, value_cat in dic_df['var_rows'].items():
            l_choice.append(value_cont['name'] + '-' + value_cat['name'])
    l_choice = ['-'] + l_choice

    return l_choice


def select_cat_eventHandler(df,
                            dic_df,
                            cdr=False):
    """
    """

    x_widget = widgets.Dropdown(
        options=list_dropdownCat(dic_df=dic_df),
        value='-',
        description='Variable:')

    return interactive(categorical,
                       {"manual": True, "auto_display": False},
                       df=fixed(df),
                       dic_multiple=fixed(dic_df),
                       variables=x_widget,
                       ca=False,
                       sheetID='',
                       sheetName='',
                       folder='',
                       move_to_drive=False,
                       move_to_drive_batch=False,
                       cdr=fixed(cdr),
                       verbose=fixed(True))

#############################################################################
#############################################################################
#############################################################################
##################### Fixed effect ##############


def focus_fixed_effect(df, group1, group2):
    """
    This function returns two graphs.
    The first one is a dot plot between the sorted continuous variable and
    the high_dimentional variable.
    The second graph counts the number of observation for each pair g1 and
    g2.
    The darker the dot, the more observations
    """

    g1 = group1
    g2 = group2

    g1_count = df.groupby([g1])[g2].nunique().sort_values(
        ascending=False).reset_index()
    g1_df = g1_count[g1]
    g1_count = g1_count.sort_values(by=g2, ascending=True)
    g2_df = pd.Series(df[g2].unique(), name=g2).sort_values()
    all_combination = pd.DataFrame(list(product(g1_df, g2_df)),
                                   columns=[g1, g2])
    list_df_combintation = []
    z = []
    fig = make_subplots(rows=2, cols=1)
    for i, name in enumerate(g1_df):
        serie_x = df[df[g1]
                     == name]
        count_hs = serie_x.groupby([g2, g1])[g2].count()
        count_hs = pd.merge(pd.DataFrame({'count': count_hs}).reset_index(),
                            all_combination[all_combination[g1] == name],
                            on=[g2, g1],
                            how='right').fillna(0).sort_values(by=g2)
        list_df_combintation.append(count_hs)
        z.append(np.array(count_hs['count']))

        # Plot dot
    trace1 = {"x": g1_count[g2],
              "y": g1_count[g1],
              "marker": {"color": "blue", "size": 10},
              "mode": "markers",
                      "name": g2,
                      "type": "scatter"
              }
    data = [trace1]
    fig.append_trace(trace1, 1, 1)
    fig['layout']['xaxis1'].update(title=g2, showgrid=False)
    fig['layout']['yaxis1'].update(title=g1, showgrid=False)
    # Plot heat map
    trace = go.Heatmap(z=np.transpose(z),
                       x=g1_df,
                       y=g2_df)
    #data = [trace]
    fig.append_trace(trace, 2, 1)
    fig['layout'].update(height=800, width=800, autosize=False,
                         title='Group fixed effect comparison count')
    # layout = {"title": "Count",
    #          "xaxis": {"title": "Count of industry", },
    #          "yaxis": {"title": "Cities"}}
    print("The maximum number of unique", g2, "is", len(g2_df))

    result = py.iplot(fig)
    return result


def select_fe_eventHandler(df):
    """
    """

    l_filter = list(df.select_dtypes(include='object'))
    l_filter = [False] + l_filter

    return interactive(focus_fixed_effect,
                       {"manual": True, "auto_display": False},
                       df=fixed(df),
                       group1=l_filter,
                       group2=l_filter)

#############################################################################
#############################################################################
#############################################################################
##################### Tab Widget ##############


# Define outputs Elements
wid_quick = widgets.Output()
# two new tab
wid_gridshow = widgets.Output()
wid_pivottable = widgets.Output()

wid_ts = widgets.Output()
wid_cont = widgets.Output()
wid_high = widgets.Output()
wid_highLow = widgets.Output()
wid_scatter = widgets.Output()
wid_scatter1 = widgets.Output()
wid_scatter2 = widgets.Output()
wid_cat = widgets.Output()
wid_fe = widgets.Output()
# Define tab widgets


def PyAnalysis(dataframe, automatic=True, Date=False, cdr=False,
               dic_ts=False, dic_Low=False, dic_high=False,
               dic_HighLow=False, dic_scatter=False, dic_scatterg1=False,
               dic_scatterg2=False, dic_cat=False):
    """
    This function launch the dashboard
    It includes all the interactive function defined above.
    it returns a tab widgets
    """
    wid_quick.clear_output()
    wid_gridshow.clear_output()
    wid_pivottable.clear_output()
    wid_ts.clear_output()
    wid_cont.clear_output()
    wid_high.clear_output()
    wid_highLow.clear_output()
    wid_scatter.clear_output()
    wid_scatter1.clear_output()
    wid_scatter2.clear_output()
    wid_cat.clear_output()
    wid_fe.clear_output()

    if automatic:
        dic_ts = create_all_keys(dataframe, date=Date, method=1)
        dic_Low = create_all_keys(dataframe, date=Date, method=2)
        dic_high = create_all_keys(dataframe, date=Date, method=3)
        dic_HighLow = create_all_keys(dataframe, date=Date, method=4)
        dic_scatter = create_all_keys(dataframe, date=Date, method=5)
        dic_scatterg1 = create_all_keys(dataframe, date=Date, method=6)
        dic_cat = create_all_keys(dataframe, date=Date, method=7)

    with wid_quick:
        display(make_quickstart(df=dataframe, cdr=cdr))

    with wid_gridshow:
        display(grid_search(df=dataframe))

    with wid_pivottable:
        display(pivot_table(df=dataframe))

    with wid_ts:
        display(select_TS_eventHandler(df=dataframe,
                                       dic_df=dic_ts,
                                       cdr=cdr))

    with wid_cont:
        display(select_catLow_eventHandler(df=dataframe,
                                           dic_df=dic_Low,
                                           cdr=cdr))

    with wid_high:
        display(select_catHigh_eventHandler(df=dataframe,
                                            dic_df=dic_high, cdr=cdr))

    with wid_highLow:
        display(select_catHighLow_eventHandler(df=dataframe,
                                               dic_df=dic_HighLow, cdr=cdr))

    with wid_scatter:
        display(select_scatter_eventHandler(df=dataframe,
                                            dic_df=dic_scatter, cdr=cdr))

    with wid_scatter1:
        display(select_scatterGroup_eventHandler(df=dataframe,
                                                 dic_df=dic_scatterg1, cdr=cdr))

    with wid_scatter2:
        display(select_scatterGroup2_eventHandler(df=dataframe,
                                                  dic_df=dic_scatterg1, cdr=cdr))

    with wid_cat:
        display(select_cat_eventHandler(df=dataframe,
                                        dic_df=dic_cat, cdr=cdr))

    with wid_fe:
        display(select_fe_eventHandler(df=dataframe))

    tab = widgets.Tab([wid_gridshow, wid_quick, wid_pivottable,
                       wid_ts, wid_cont, wid_high,
                       wid_highLow, wid_scatter, wid_scatter1, wid_scatter2, wid_cat, wid_fe])

    tab.set_title(0, 'Filter dataset')
    tab.set_title(1, 'Quick description')
    tab.set_title(2, 'Pivot table')
    tab.set_title(3, 'Time Serie plots')
    tab.set_title(4, 'Low dimensional group')
    tab.set_title(5, 'High dimensional group')
    tab.set_title(6, 'High/Low dimensional group')
    tab.set_title(7, 'Scatter plot')
    tab.set_title(8, 'Scatter plot, group 1')
    tab.set_title(9, 'Scatter plot, group 2')
    tab.set_title(10, 'Categorical analysis')
    tab.set_title(11, 'Fixed effect')

    display(tab)
