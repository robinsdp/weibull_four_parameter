# Assumes working directory is at /.../weibull_four_parameter/
# Example of using code in python
from calc_4_param_weibull import *

weib_1 = weibull_4_parameter()
weib_1.load_values_to_fit("example/token_data.txt")
mle1 = weib_1.calc_ideal_parameters()
weib_1.print_parameters()

# Create a list of values on which to form predictions, potentially for plotting
incrementor = .05
vals_to_predict = [round(x*incrementor,3) for x in range(int(-3/incrementor), int(2.5/incrementor))]
vals_to_predict = [10**x for x in vals_to_predict]

# Get predictions from the weibul distribution
predictions = weib_1.predict_4_param_weibul_cdf(vals_to_predict)
# Export predictions to a file.
weib_1.export_4_param_weibul_cdf_preds("weibull_pred_values.txt", vals_to_predict)

# Alternatively, just call the function 'fit_and_export' if we just want the fit parameters:
fit_and_export("example/token_data.txt", negative_double_log = True)

# Also provide values to get predictions on and export those to a file location
fit_and_export("example/token_data.txt", output_location = "weibull_pred_values.txt", prediction_values = 'example/plot_val_at.txt', negative_double_log = True)
