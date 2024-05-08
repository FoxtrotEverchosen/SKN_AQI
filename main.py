import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns


df = pd.read_excel('DATA_CLEAN.xlsx')

date_time = pd.to_datetime(df.pop('data'), format='%Y%m%d')

# indeksy kolumn
column_indices = {name: i for i, name in enumerate(df.columns)}
n = len(df)

# podzial danych: 70% - trenowanie, 15% - walidacja, 15% - testowanie
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7): int(n*0.85)]
test_df = df[int(n*0.85):]

num_features = df.shape[1]

# normalizacja danych wejsciowych
train_mean = train_df.mean()
train_standard_deviation = train_df.std()

train_df = (train_df - train_mean) / train_standard_deviation
val_df = (val_df - train_mean) / train_standard_deviation
test_df = (test_df - train_mean) / train_standard_deviation

# podglad dystrybucji danych wejsciowych
df_std = (df - train_mean) / train_standard_deviation
df_std = df_std.melt(var_name="Kolumna", value_name="Dystrybucja")
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x="Kolumna", y="Dystrybucja", data=df_std)
_ = ax.set_xticklabels(df.keys(), rotation=90)
# plt.show()


class WindowGenerator:
    def __init__(self, input_width, label_width, shift,
                 train_df=train_df, val_df=val_df, test_df=test_df,
                 label_columns=None):

        # Dane do trenowania modelu sa narzucone z gory
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # ustalenie indeksow dla kolumn w oknie
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Parametry okna
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width, 1)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None, 1)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]

        if self.label_columns is not None:
            labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    # not needed anymore!
    def plot(self, model=None, plot_col='PM10 ug/m3', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'PM10 [normalizowany]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Dane wejściowe', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Wartości prawdziwe', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Wartości przewidywane',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Czas [dni]')

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,
        )

        ds = ds.map(self.split_window)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result


# w1 = WindowGenerator(input_width=24, label_width=1, shift=24, label_columns=['temp'])
# w2 = WindowGenerator(input_width=6, label_width=1, shift=1, label_columns=['temp'])
# print(w1, w2)

"""
# Stack three slices, the length of the total window.
example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
                           np.array(train_df[100:100+w2.total_window_size]),
                           np.array(train_df[200:200+w2.total_window_size])])

example_inputs, example_labels = w2.split_window(example_window)


print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}')
"""

single_step_window = WindowGenerator(input_width=1, label_width=1, shift=1, label_columns=["PM10 ug/m3"])

"""
for example_inputs, example_labels in single_step_window.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')
"""


class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]


baseline = Baseline(label_index=column_indices['PM10 ug/m3'])
baseline.compile(loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanAbsoluteError()])
val_performance = {}
performance = {}
val_performance['Bazowy'] = baseline.evaluate(single_step_window.val, return_dict=True)
performance['Bazowy'] = baseline.evaluate(single_step_window.test, verbose=0, return_dict=True)

wide_window = WindowGenerator(input_width=24, label_width=24, shift=1, label_columns=['PM10 ug/m3'])
print(wide_window)

wide_window.plot(baseline)
plt.suptitle("Model Bazowy")
plt.show()


linear = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1)
])
print('Input shape:', single_step_window.example[0].shape)
print('Output shape:', linear(single_step_window.example[0]).shape)


MAX_EPOCHS = 20


def compile_and_fit(model, window, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(), metrics=[tf.keras.metrics.MeanAbsoluteError()])
    history = model.fit(window.train, epochs=MAX_EPOCHS, validation_data=window.val, callbacks=[early_stopping])
    return history


history = compile_and_fit(linear, single_step_window)

val_performance['Liniowy'] = linear.evaluate(single_step_window.val, return_dict=True)
performance['Liniowy'] = linear.evaluate(single_step_window.test, verbose=0, return_dict=True)

wide_window.plot(linear)
plt.suptitle("Model Liniowy")
plt.show()
plt.bar(x=range(len(train_df.columns)),
        height=linear.layers[0].kernel[:, 0].numpy())
axis = plt.gca()
axis.set_xticks(range(len(train_df.columns)))
_ = axis.set_xticklabels(train_df.columns, rotation=90)
plt.show()

dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation="relu"),
    tf.keras.layers.Dense(units=64, activation="relu"),
    tf.keras.layers.Dense(units=1)
])
history = compile_and_fit(dense, single_step_window)

val_performance['Sieć Neuronowa'] = dense.evaluate(single_step_window.val, return_dict=True)
performance['Sieć Neuronowa'] = dense.evaluate(single_step_window.test, verbose=0, return_dict=True)
wide_window.plot(dense)
plt.suptitle("Prosta sieć nueronowa")
plt.show()

CONV_WIDTH = 3
conv_window = WindowGenerator(input_width=CONV_WIDTH, label_width=1, shift=1, label_columns=['PM10 ug/m3'])

conv_window.plot()
plt.suptitle('Given values of 3 days as inputs, predict 1 day into the future')
plt.show()

multi_step_dense = tf.keras.Sequential([
    # Shape: (time, features) => (time*features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation="relu"),
    tf.keras.layers.Dense(units=32, activation="relu"),
    tf.keras.layers.Dense(units=1),
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    tf.keras.layers.Reshape([1, -1])
])

print('Input shape:', conv_window.example[0].shape)
print('Output shape:', multi_step_dense(conv_window.example[0]).shape)

history = compile_and_fit(multi_step_dense, conv_window)
val_performance['Multi Step Dense'] = multi_step_dense.evaluate(conv_window.val, return_dict=True)
performance['Multi Step Dense'] = multi_step_dense.evaluate(conv_window.test, verbose=0, return_dict=True)

conv_window.plot(multi_step_dense)
plt.show()

conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=(CONV_WIDTH,), activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

print("Conv model on `conv_window`")
print('Input shape:', conv_window.example[0].shape)
print('Output shape:', conv_model(conv_window.example[0]).shape)

history = compile_and_fit(conv_model, conv_window)
val_performance['CNN'] = conv_model.evaluate(conv_window.val, return_dict=True)
performance['CNN'] = conv_model.evaluate(conv_window.test, verbose=0, return_dict=True)

LABEL_WIDTH = 24
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH-1)
wide_conv_window = WindowGenerator(
    input_width=INPUT_WIDTH,
    label_width=LABEL_WIDTH,
    shift=1,
    label_columns=['PM10 ug/m3']
)

print(wide_conv_window)
wide_conv_window.plot(conv_model)
plt.suptitle("CNN")
plt.show()

lstm_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dense(units=1)
])

history = compile_and_fit(lstm_model, wide_window)
val_performance['LSTM'] = lstm_model.evaluate(wide_window.val, return_dict=True)
performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0, return_dict=True)
wide_window.plot(lstm_model)
plt.suptitle("LSTM")
plt.show()

cm = lstm_model.metrics[1]
x = np.arange(len(performance))
width = 0.3
metric_name = 'mean_absolute_error'
val_mae = [v[metric_name] for v in val_performance.values()]
test_mae = [v[metric_name] for v in performance.values()]

plt.ylabel('średni błąd bezwzględny [PM10, normalizowany]')
plt.bar(x - 0.17, val_mae, width, label='Walidacja')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=performance.keys(),
           rotation=45)
_ = plt.legend()
plt.show()