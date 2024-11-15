
import json
from datetime import datetime
import torch
import pickle
import pytz
import scripts.Simulation_config as config
import scripts.Model
from collections import defaultdict
from itertools import islice
from pprint import pprint
import numpy as np
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.colors as mcolors
import calendar

# Persist dicts with complex keys.
# The dict keys are converted from multi-class into json format.
#
class Serialize:

    # Use torch to save a dictionary with training results to disc.
    # Especially useful for torch models.
    #
    @staticmethod
    def store_results_with_torch(all_trained_models):      
        
        # Squeeze the dict keys
        all_trained_models = Serialize.get_serialized_dicts(all_trained_models, isModel = True)

        # Save the total model with torch.save
        timestamp = Serialize.get_act_timestamp()
        torch.save(all_trained_models, f'scripts/outputs/all_trained_models{timestamp}.pth')
        torch.save(all_trained_models, f'scripts/outputs/all_trained_models.pth')

    # Use pickle to save a dictionary with training results to disc.
    #
    @staticmethod
    def store_results_with_pickle(all_train_histories):
        
        # Squeeze the dict keys
        all_train_histories = Serialize.get_serialized_dicts(all_train_histories, isModel = False)
        
        # Store the variables in a persistent files with the timestamp
        timestamp = Serialize.get_act_timestamp()
        with open(f"scripts/outputs/all_train_histories{timestamp}.pkl", 'wb') as f:
            pickle.dump(all_train_histories, f)
        with open(f"scripts/outputs/all_train_histories.pkl", 'wb') as f:
            pickle.dump(all_train_histories, f)

    # Serialize the given dicts.
    #
    @staticmethod
    def get_serialized_dicts(dict_with_komplex_keys, isModel):
        serialized_models = {}
        for key, data in dict_with_komplex_keys.items():
            serialized_key = Serialize.serialize_complex_key(key)
            if isModel:
                # Not the whole models shall be saved, but only its parameters.
                model = data
                serialized_models[serialized_key] = model.state_dict()
            else:
                serialized_models[serialized_key] = data

        return serialized_models
    
    # Change the complex key of the given train history dictionary to a simple json key.
    #
    @staticmethod
    def serialize_complex_key(complex_key):

        # Serialize the config object by iterating over the fields and handling each type appropriately
        config_serialized = {field: Serialize.serialize_value(getattr(complex_key[2], field)) for field in complex_key[2]._fields}

        # Convert the whole key to a json
        string_key = json.dumps({
            "model_type": complex_key[0],
            "load_profile": complex_key[1],
            "config": config_serialized
        })
        return string_key

    # Helper function for the serialization. Return the value of the named tuple field.
    #
    @staticmethod
    def serialize_value(value):
        if isinstance(value, (bool, int, str, tuple)):  # Basic types
            return value
        elif isinstance(value, type):
            return value.__dict__  # If the object is a class, etc.
        else:
            assert False, "Unimmplemented variable occurs in the config."

    # Return a current timestamp.
    #
    @staticmethod
    def get_act_timestamp(tz='Europe/Vienna'):
        return datetime.now(pytz.timezone(tz)).strftime("_%Y%m%d_%H%M")

# Get the trained models and train history from disc.
#
class Deserialize:

    # Get the training histories from disc and deserialize/unpack them.
    #
    @staticmethod
    def get_training_histories(path):
        
        with open(path, 'rb') as file:
            train_histories_serialized = pickle.load(file)
        
        train_histories = {}        
        for serialized_key, history in train_histories_serialized.items():
            deserialize_key = Deserialize.deserialize_key(serialized_key)            
            train_histories[deserialize_key] = history
        
        return train_histories
    
    # Get the trained models from disc and deserialize/unpack them.
    #
    @staticmethod
    def get_trained_model(path_to_trained_parameters, model_type, test_profile, chosenConfig, num_of_features):
        
        serialized_dict = torch.load(path_to_trained_parameters)
        
        for serialized_key, state_dict in serialized_dict.items():
            deserialize_key = Deserialize.deserialize_key(serialized_key)
            if model_type == deserialize_key[0] and \
                test_profile == deserialize_key[1] and \
                chosenConfig == deserialize_key[2]:
                model = scripts.Model.Model(model_type=deserialize_key[0], 
                                            model_size=deserialize_key[2].modelSize,
                                            num_of_features=num_of_features,
                                            )
                model.my_model.load_state_dict(state_dict)
                return model
        
        assert False, "Model not found!"

    # Convert a dict to a named tuple
    #
    @staticmethod
    def dict_to_named_tuple(**kwargs):
        return config.Config_of_one_run(**kwargs)

    # If the named tuple includes lists, convert it to a tuples.
    #
    @staticmethod
    def convert_lists_to_tuples(config_namedtuple):
        config_dict = config_namedtuple._asdict()
        for field, value in config_dict.items():
            if isinstance(value, list):
                config_dict[field] = tuple(value)
        return type(config_namedtuple)(**config_dict)

    # Helper function to deserialize keys and recreate the config namedtuple
    #
    @staticmethod
    def deserialize_key(serialized_key):

        # Rebuild named tuple from the dictionary
        key_dict = json.loads(serialized_key)
        config_rebuilt_named_tuple = Deserialize.dict_to_named_tuple(**key_dict['config'])
        config_rebuilt = Deserialize.convert_lists_to_tuples(config_rebuilt_named_tuple)

        return (
            key_dict['model_type'],
            key_dict['load_profile'],
            config_rebuilt
        )

# Evaluate the stored training results 
#
class Evaluate_Models:

    # Get all training results as nested dicts
    #
    @staticmethod
    def get_training_results(path_to_train_histories, 
                             skip_first_n_configs=None, 
                             skip_last_n_configs=None,                              
                             ):
        
        all_train_histories = Deserialize.get_training_histories(path_to_train_histories)

        # Create a nested dictionary of the results
        result_per_config = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
        for key, value in all_train_histories.items():
            model_type = key[0]
            load_profile = key[1]
            sim_config = key[2]
            results = value
            result_per_config[sim_config][model_type][load_profile]['loss'] = float(results['loss'][-1])
            result_per_config[sim_config][model_type][load_profile]['test_loss'] = float(results['test_loss'][-1])
            result_per_config[sim_config][model_type][load_profile]['test_sMAPE'] = float(results['test_sMAPE'][-1])

        # Optionally: Skip given configs
        result_per_config = dict(islice(result_per_config.items(), skip_first_n_configs, skip_last_n_configs))

        return result_per_config

    # Get the best models per energy community (i.e. the "winners")
    #
    @staticmethod
    def get_winner_models(path_to_train_histories, do_print=False):
        
        result_per_config = Evaluate_Models.get_training_results(path_to_train_histories)
        
        # Calculate the best model ("winner") for each profile
        winner_per_config = {}
        winner_per_model = {}
        for config, result_per_model in result_per_config.items():
                
            # Initialize the used dict values with 0
            winner_per_config[config] = {model_type: 0 for model_type in result_per_model.keys()}
            for model_type in result_per_model:
                winner_per_model.setdefault(model_type, 0)
                
            # Loop through each load_profile within the model data
            for load_profile in next(iter(result_per_model.values())).keys():
                
                # Determine the model type with the best performance for the current load profile
                best_model = min(result_per_model, key = lambda model: result_per_model[model][load_profile]['test_sMAPE'])
                
                # Increment the count for the winning model type
                winner_per_config[config][best_model] += 1
                winner_per_model[best_model] += 1

        # Summarize the wins per model
        #
        if do_print:
            print(f'Total Winner per model:\n  Models:', end=" ")
            counts = ""
            for model, count in winner_per_model.items():
                print(f'{model}', end=" ")
                counts += f'& {count} '
            print(f'{  counts}')
        
        return winner_per_config, winner_per_model

    # Calculate and print results for each config and model_type
    #
    @staticmethod
    def print_results(path_to_train_histories, print_style = 'pandas_df'):
        
        result_per_config = Evaluate_Models.get_training_results(path_to_train_histories)
        result_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        for config, result_per_model in result_per_config.items():
            
            if print_style != 'pandas_df':
                pprint(f'Configuration: {config}')            
            
            for model_type, result_per_profile in result_per_model.items():
                
                # Get al list of all losses of the current config and modeltype
                train_losses, test_MAE, test_sMAPE  = [], [], []
                for load_profile, results in result_per_profile.items():
                    train_losses.append(results['loss'])
                    test_MAE.append(results['test_loss'])
                    test_sMAPE.append(results['test_sMAPE'])        
                assert len(result_per_profile) == config.nrOfComunities
                
                decimal_points_MAE = 4
                decimal_points_sMAPE = 2
                mean_test_MAE = f'{np.mean(test_MAE):.{decimal_points_MAE}f}'
                mean_test_sMAPE = f'{np.mean(test_sMAPE):.{decimal_points_sMAPE}f}'
                std_dev_test_MAE = f'{np.std(test_MAE):.{decimal_points_MAE}f}'
                std_dev_test_sMAPE = f'{np.std(test_sMAPE):.{decimal_points_sMAPE}f}'
                mean_train_MAE = f'{np.mean(train_losses):.{decimal_points_MAE}f}'

                if print_style == 'pandas_df':
                    result_dict[config][model_type] = test_sMAPE
                elif print_style == 'shell':
                    # Print the results of the current config and modeltype
                    print(f'    Model: {model_type}')
                    print(f'      Mean Test MAE: {mean_test_MAE}')
                    print(f'      Mean Test sMAPE: {mean_test_sMAPE}')
                    print(f'      Standard Deviation Test MAE: {std_dev_test_MAE}')
                    print(f'      Standard Deviation Test sMAPE: {std_dev_test_sMAPE}')
                    print(f'      Mean Train MAE: {mean_train_MAE}\n')
                else:
                    assert "Please choose correct 'print_style' argument."
        
        return result_dict

    # Calculate and print results for each config and model_type
    #
    @staticmethod
    def plot_training_losses_over_epochs(path_to_train_histories, 
                                         plot_only_single_config = False,
                                         plotted_config = None):
        
        all_train_histories = Deserialize.get_training_histories(path_to_train_histories)

        # Target config(s) to plot
        if plot_only_single_config:
            print(f"Plotted Config:")
            pprint(f"{plotted_config}")
        filtered_train_histories = dict(
            (key, value) for key, value in all_train_histories.items() 
            if plot_only_single_config == False or key[2] == plotted_config
        )

        # Create a combined list of loss values and corresponding run names
        combined_loss_data = []
        for run_id, (run_config, history) in enumerate(filtered_train_histories.items()):
            # Define the labels of the following graph
            model_type, load_profile, act_config = run_config
            label = (model_type, act_config.aggregation_Count, act_config.modelSize)

            # Add each epoch's loss for the current run's history
            for epoch, loss_value in enumerate(history['loss']):
                combined_loss_data.append((f"{label}_{run_id}", epoch + 1, loss_value))

        # Create a DataFrame from the combined loss data
        df = pd.DataFrame(combined_loss_data, columns=['Run_History', 'Epoch', 'Loss'])

        # Use plotly express to plot the line graph
        fig = px.line(
            df, 
            x='Epoch', 
            y='Loss', 
            color='Run_History', 
            line_group='Run_History',
            labels={'Loss': 'Training Loss', 'Epoch': 'Epoch'},
            color_discrete_sequence=px.colors.sequential.Blues,
        )

        # Customize axes and grid
        fig.update_yaxes(
            showline=True, 
            linewidth=1, 
            linecolor='black', 
            mirror=True,
            gridcolor='lightgrey',
            gridwidth=0.5
        )
        fig.update_xaxes(
            showline=True, 
            linewidth=1, 
            linecolor='black', 
            mirror=True,
            gridcolor='lightgrey',
            gridwidth=0.5
        )

        # Additional layout customizations
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(
            showlegend=False, 
            plot_bgcolor='white',
            width=1000,
            height=500
        )

        # Save and show the plot
        fig.write_image('scripts/outputs/figs/loss_over_epochs.pdf', format='pdf')
        fig.show()


    @staticmethod
    def create_calendar_plot(startdate, daily_values):
        """
        Creates a calendar-style heatmap plot displaying daily values starting from a specified date.

        Parameters:
        ----------
        startdate : str or datetime
            The initial date for the plot, used to arrange daily values in calendar order.

        daily_values : array-like
            A sequence of values representing data for each day, to be visualized in the calendar.

        Returns:
        -------
        None
            Displays a calendar plot with each month organized in weekly rows, coloring each day based on its value.
            Saves the plot as a borderless PDF.
        """
    
        # Create a DataFrame for daily data
        dates = pd.date_range(start=startdate, periods=len(daily_values), freq='D')
        daily_df = pd.DataFrame({'date': dates, 'value': daily_values})

        # Add additional columns for year, month, day, and weekday
        daily_df['year'] = daily_df['date'].dt.year
        daily_df['month'] = daily_df['date'].dt.month
        daily_df['day'] = daily_df['date'].dt.day
        daily_df['weekday'] = daily_df['date'].dt.weekday  # Monday = 0, Sunday = 6

        # Get unique years and months in the data
        years = daily_df['year'].unique()
        months = daily_df['month'].unique()
        num_months = len(months)

        # Determine the grid size dynamically based on the number of months
        rows = (num_months + 2) // 3  # Arrange in 3 columns

        # Set up the figure and axis grid for the required months
        fig, axs = plt.subplots(rows, 3, figsize=(16, rows * 4), constrained_layout=False)

        # Adjust the spacing between rows and columns
        plt.subplots_adjust(hspace=0.3, wspace=0.1)

        # Chose colors
        cmap = plt.cm.Blues
        norm = mcolors.Normalize(vmin=daily_df['value'].min(), vmax=daily_df['value'].max())

        # Flatten the axis array for easy iteration, considering any unused subplots
        axs = axs.flatten()

        # Iterate through each unique month and year in the data
        plot_idx = 0
        for year in sorted(years):
            for month in range(1, 13):
                monthly_data = daily_df[(daily_df['year'] == year) & (daily_df['month'] == month)]
                if monthly_data.empty:
                    continue  # Skip months with no data

                # For the first month with data, set up the start day and row appropriately
                if month == daily_df['month'].iloc[0] and year == daily_df['year'].iloc[0]:
                    start_day = monthly_data['weekday'].iloc[0]  # Starting weekday
                    starting_day_in_month = monthly_data['day'].iloc[0]  # Starting day
                    start_row = (start_day + starting_day_in_month - 1) // 7  # Calculate starting row
                else:
                    start_day = calendar.monthrange(year, month)[0]  # First weekday of the month (0=Monday)
                    starting_day_in_month = 1  # Start from the first day for subsequent months
                    start_row = 0  # Start at the top row for full months

                days_in_month = calendar.monthrange(year, month)[1]  # Number of days in the month

                # Create a grid for each month, initializing empty cells with NaN for alignment
                month_matrix = np.full((6, 7), np.nan)  # Max of 6 weeks per month, 7 days per week

                # Fill in day values in the grid, starting from the appropriate row and day in the first month
                day = starting_day_in_month
                for day_idx in range(start_day, start_day + days_in_month):
                    row = start_row + (day_idx // 7)
                    col = day_idx % 7
                    day_value = monthly_data[monthly_data['day'] == day]['value'].values
                    if day_value.size > 0:
                        month_matrix[row, col] = day_value[0]
                    day += 1

                # Get the subplot axis for the current month
                ax = axs[plot_idx]
                plot_idx += 1
                ax.set_title(f"{calendar.month_name[month]} {year}", fontsize=14, pad=10, color="0.3")

                # Set light grey borders around each monthly subplot
                for spine in ax.spines.values():
                    spine.set_edgecolor("lightgrey")
                    spine.set_linewidth(0.5)

                # Display the month data as a heatmap, with NaNs handled automatically
                cax = ax.imshow(month_matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

                # Annotate each cell with the day number, starting from the correct day in the first month
                day = starting_day_in_month
                for week in range(6):  # 6 rows (weeks) per month
                    for weekday in range(7):  # 7 columns (days)
                        if not np.isnan(month_matrix[week, weekday]):
                            ax.text(weekday, week, str(day), ha='center', va='center', fontsize=8, color="0.3")
                            day += 1  # Move to the next day in the month

                # Set x-axis and y-axis ticks to align the grid with daily boxes
                ax.set_xticks(np.arange(-0.5, 7, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
                ax.grid(which="minor", color="0.95", linestyle='-', linewidth=0.5)  # Grid around each daily box

                # Keep weekday labels but remove small ticks on x and y axes
                ax.tick_params(axis="both", which="both", length=0)

                # Set x-axis labels for weekdays
                ax.set_xticks(range(7))
                ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], fontsize=10, color="0.3")
                ax.set_yticks([])

        # Add a color bar for the legend
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, orientation='vertical', fraction=0.03, pad=0.04)
        cbar.set_label("sMAPE (%)", fontsize=12, color="0.3")
        cbar.outline.set_edgecolor(color="lightgrey")
        cbar.outline.set_linewidth(1)
        cbar.ax.yaxis.set_tick_params(labelcolor="0.3")

        # Hide any unused subplots
        for i in range(plot_idx, len(axs)):
            fig.delaxes(axs[i])

        plt.savefig("scripts/outputs/figs/calendar_plot.pdf", format="pdf", bbox_inches="tight", pad_inches=0.1)
        plt.show()

    @staticmethod
    def print_latex_table(result_dict, configs_to_print, config_groups, config_names):
            latex_string = ''
            decimal_points = 2

            for i, expected_config in enumerate(configs_to_print):
                    expected_config_found = False
                    if config_groups[i] != '-':
                            latex_string += '\\hline\n'
                            latex_string += f'\\multirow{{{config_groups[i][1]}}}{{*}}{{\\textbf{{{config_groups[i][0]}}}}} \n'
                    latex_string += f'    & {config_names[i]}'
                    for available_config in result_dict:
                            if expected_config == available_config:
                                    expected_config_found = True                                        
                                    
                                    # Find the best mean_test_sMAPE and its model
                                    best_mean_test_sMAPE = float('inf')
                                    best_model_type = None
                                    for model_type, test_sMAPE in result_dict[available_config].items():
                                            mean_test_sMAPE = np.mean(test_sMAPE)
                                            if mean_test_sMAPE < best_mean_test_sMAPE:
                                                    best_mean_test_sMAPE = mean_test_sMAPE
                                                    best_model_type = model_type

                                    # Print the metrics per model and mark the best model
                                    for model_type, test_sMAPE in result_dict[available_config].items():                                

                                            mean_test_sMAPE_str = f'{np.mean(test_sMAPE):.{decimal_points}f}'
                                            std_test_sMAPE_str = f'{np.std(test_sMAPE):.{decimal_points}f}'

                                            if model_type == best_model_type:
                                                    latex_string += f' & \\textbf{{{mean_test_sMAPE_str}}} ({std_test_sMAPE_str})'
                                            else:
                                                    latex_string += f' & {mean_test_sMAPE_str} ({std_test_sMAPE_str})'
                                            
                                    latex_string += ' \\\ \n'
                    if expected_config_found == False:
                            # expected config wasn't found in the test run file
                            latex_string += ' & - & - & - & - & - & - \\\ \n'
            print(latex_string)

                
        
        