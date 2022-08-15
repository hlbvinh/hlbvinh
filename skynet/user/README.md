# User Model (DRAFT)

We're building a personalized user model which for any given condition and user
should output a suggested humidex that will feel comfortable to the user.

We'll start with a baseline model which we generate from observations of the
initial users of the humidex mode (so far, Julian, Paul and myself) and their
feedback through the app.

In production the user model will be personalized over time with the feedback
from the individual users.

As features we'll use sensor data, AC state data, time of day as well as the
user_id.

As targets we want to use humidex values we think are comfortable for the
targets. In a first iteration we will simplify the problem a bit and ignore the
lag between the time when we change the AC settings and the time the conditions
stabilize.

Once we have these targets we can train any standard regressor to predict the
comfort humidex for given conditions, time of day and user. This value can then
be fed to the humidex control.

## Base Model
As base model we can simply use the the standard model with unknown user_id.

