import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
os.environ['HOST'] = '127.0.0.1'

class PlotlyApp:
    def __init__(
            self,
            X_model,
            Y_model,
            model,
            modelAdapter,
            predictions = None,
            timezone = 'UTC+00:00',
            Y_model_pretrain = None,
            modelAdapter_pretrain = None
                 ):
        
        # Initialize the Dash app
        self.app = dash.Dash(__name__)
        self.X_plot = X_model
        self.Y_plot = Y_model
        self.model_plot = model
        self.modelAdapter = modelAdapter
        self.predictions = predictions
        self.timezone = timezone
        self.Y_model_pretrain = Y_model_pretrain
        self.modelAdapter_pretrain = modelAdapter_pretrain

        # Define the layout of the app
        self.app.layout = html.Div([
            dcc.Dropdown(
                id='dataset-picker',
                options=[
                    {'label': 'train dataset', 'value': 'train'},
                    {'label': 'dev dataset', 'value': 'dev'},
                    {'label': 'test dataset', 'value': 'test'},
                    {'label': 'whole dataset - unshuffled', 'value': 'all'},
                ],
                value='all',   # default dataset
                multi=False,  # Set to True for multi-selection dropdown
                style={'width': '40%', 'fontSize': 16, 'padding': '0px', 'marginBottom': 10}
            ),

            dcc.Input(
                id='date-picker',
                type='number',
                value=1,
                min=1,
                style={'width': '7%', 'fontSize': 13, 'padding': '10px', 'marginBottom': 10}
            ),

            html.Label(id='output-label'),

            dcc.Graph(id='date-plot1'),
            dcc.Graph(id='date-plot2')
        ])

        # Callback to update the text of the HTML label
        self.app.callback(
            Output('output-label', 'children'),
            [Input('dataset-picker', 'value'), Input('date-picker', 'value')]
        )(self.update_label)

        # Define callback to update the graph based on the selected date
        self.app.callback(
            Output('date-plot1', 'figure'),
            Output('date-plot2', 'figure'),
            [Input('dataset-picker', 'value'), Input('date-picker', 'value')]
        )(self.update_date_plot)

    def update_label(self, selected_dataset, selected_date):
            
        # Validate the inputs
        #
        if selected_dataset == None:
            selected_dataset = 'test'
        if selected_date == None:
            selected_date = 1
        elif selected_date >= self.Y_plot[selected_dataset].shape[0] - 1:
            selected_date = self.Y_plot[selected_dataset].shape[0] - 1

        # Get the currently choosen data subset (train, dev or test)
        if selected_dataset == 'all':
            subset = self.modelAdapter.getDatasetTypeFromIndex(selected_date)
            subset_text = f" Subset: {subset}."
        else:
            subset_text = ""

        # Convert the one-hot weekday encoding to a short string representation
        prediction_timestep = int(self.modelAdapter.prediction_history.total_seconds() / (60.0 * 60.0))
        weekday_one_hot = self.X_plot[selected_dataset][selected_date, prediction_timestep, :7]
        weekday_str = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][np.argmax(weekday_one_hot)]

        # Display information about the currently chosen date
        available_days = self.Y_plot[selected_dataset].shape[0] - 1
        returnValue = f"   ... selected day from [1 ... {available_days}]. Weekday of the prediction timestep: {weekday_str}." + subset_text

        return returnValue
    
    def update_date_plot(self, selected_dataset, selected_date):

        try: # use a try-catch to prevent a kernel crash

            # Validate the inputs
            #
            if selected_dataset == None:
                selected_dataset = 'all'
            if selected_date == None:
                selected_date = 1
            elif selected_date >= self.Y_plot[selected_dataset].shape[0] - 1:
                selected_date = self.Y_plot[selected_dataset].shape[0] - 1

            # Get the real measured power profile of the selected day
            Y_real = self.Y_plot[selected_dataset][selected_date,:,0]
            Y_real = self.modelAdapter.deNormalizeY(Y_real)

            # Get the predicted power profile of the selected day
            X_selected = self.X_plot[selected_dataset]
            if self.predictions is not None:
                Y_pred = self.predictions[selected_date][0,:,0]
                if selected_dataset != 'all':
                    print("Warning: Without given model, the visualiation only works for the 'all' dataset", flush=True)
            else:
                Y_pred = self.model_plot.predict(X_selected)
                Y_pred = Y_pred[selected_date,:,0]
            Y_pred = self.modelAdapter.deNormalizeY(Y_pred)

            # Create a DataFrame for Plotly Express
            startdate = self.modelAdapter.getStartDateFromIndex(selected_dataset, selected_date)
            datetime_index = pd.date_range(start=startdate, periods=Y_pred.shape[0], freq='1h').tz_convert(self.timezone)

            if self.Y_model_pretrain is None:
                df_Y = pd.DataFrame({'x': datetime_index, 'Y_real': Y_real, 'Y_pred': Y_pred})
            else:
                # Add scaled standard load profile
                Y_standardload_denormalized = self.modelAdapter_pretrain.deNormalizeY(self.Y_model_pretrain[selected_dataset][selected_date,:,0])
                df_Y = pd.DataFrame({'x': datetime_index, 'Y_real': Y_real, 'Y_pred': Y_pred, 'Y_standardload': Y_standardload_denormalized})

            # Create a line chart using Plotly Express
            fig_Y = px.line()
            fig_Y.add_scatter(x=df_Y['x'], y=df_Y['Y_real']/1000.0, mode='lines', name='Real', line_color='lightgrey')
            fig_Y.add_scatter(x=df_Y['x'], y=df_Y['Y_pred']/1000.0, mode='lines', name='Predicted', line_color='blue')
            fig_Y.update_layout(yaxis_title='Load Profile (kW)', xaxis_title='Time (HH:MM)', 
                                plot_bgcolor='white', legend=dict(x=0, y=1, xanchor='left', yanchor='top'),
                                margin=dict(l=20, r=20, t=20, b=20),
                                )
            fig_Y.update_xaxes(showline = True, linewidth = 1, linecolor = 'black', )
            fig_Y.update_yaxes(showline = True, linewidth = 1, linecolor = 'black', )
            fig_Y.update_xaxes(showline = True, linewidth = 1, linecolor = 'black', mirror = True)
            fig_Y.update_yaxes(showline = True, linewidth = 1, linecolor = 'black', mirror = True)
            if self.Y_model_pretrain is not None:
                fig_Y.add_scatter(x=df_Y['x'], y=df_Y['Y_standardload']/1000.0, mode='lines', name='Y_standardload')

            # # Additionally visualize the input Data of the LSTM
            # # Create a dataframe
            startdate = self.modelAdapter.getStartDateFromIndex(selected_dataset, selected_date) - self.modelAdapter.prediction_history
            datetime_index = pd.date_range(start=startdate, periods=X_selected.shape[1], freq='1h')
            X_visualized = X_selected[selected_date,:,:]
            df_X = pd.DataFrame(X_visualized, index=datetime_index)

            # Create a figure with subplots and shared x-axis
            fig_X = make_subplots(rows=df_X.shape[1], cols=1, shared_xaxes=True, subplot_titles=df_X.columns)
            for i, column in enumerate(df_X.columns):
                fig_X.add_trace(go.Scatter(x=df_X.index, y=df_X[column], mode='lines', name=column), row=i+1, col=1)
            fig_X.update_layout(
                                #yaxis_title='LSTM inputs', 
                                height=1200, 
                                plot_bgcolor='white', showlegend=False,
                                #yaxis_title_shift=-50, yaxis_title_standoff=0
                                )
            fig_X.update_xaxes(showgrid=True, gridcolor='lightgrey')
            fig_X.update_yaxes(showgrid=True, gridcolor='lightgrey')

            # Store the create figure
            fig_Y.write_image('scripts/outputs/plotly_profile_Y.pdf', format='pdf')
            fig_X.write_image('scripts/outputs/plotly_profile_X.pdf', format='pdf')

            return fig_Y, fig_X
        
        except:
            print("hoi!")
            return px.line(), px.line()
        

    def run(self, myport=8050):
        # Run the app
        self.app.run(debug=True, port=myport)

if __name__ == '__main__':
    lstm_app = PlotlyApp()
    lstm_app.run()
