# Polars TI - A Technical Indicators Library in Python 3

[![license](https://img.shields.io/github/license/CMobley7/polars-ti)](#license)
[![Stars](https://img.shields.io/github/stars/CMobley7/polars-ti?style=flat)](#stars)
[![Forks](https://img.shields.io/github/forks/CMobley7/polars-ti?style=flat)](#forks)
[![Contributors](https://img.shields.io/github/contributors/CMobley7/polars-ti?style=flat)](#contributors)
[![Issues](https://img.shields.io/github/issues-raw/CMobley7/polars-ti?style=flat)](#issues)
[![Closed Issues](https://img.shields.io/github/issues-closed-raw/CMobley7/polars-ti?style=flat)](#closed-issues)

_Polars Technical Indicators_ (**Polars TI**) is an easy to use library that leverages the Pandas package with more than 150+ Indicators and Utility functions and more than 60+ TA-Lib Candlestick Patterns. Many commonly used indicators are included, such as: _Candle Pattern_(**cdl_pattern**), _Simple Moving Average_ (**sma**) _Moving Average Convergence Divergence_ (**macd**), _Hull Exponential Moving Average_ (**hma**), _Bollinger Bands_ (**bbands**), _On-Balance Volume_ (**obv**), _Aroon & Aroon Oscillator_ (**aroon**), _Squeeze_ (**squeeze**) and **_many more_**.

**Note:** _TA Lib_ must be installed to use **all** the Candlestick Patterns. `pip install TA-Lib`. If _TA Lib_ is not installed, then only the builtin Candlestick Patterns will be available.

<br/>

# **Table of contents**

<!--ts-->

- [Features](#features)
- [Installation](#installation)
  - [Stable](#stable)
  - [Latest Version](#latest-version)
  - [Cutting Edge](#cutting-edge)
- [Quick Start](#quick-start)
- [Help](#help)
- [Issues and Contributions](#issues-and-contributions)
- [Programming Conventions](#programming-conventions)
  - [Standard](#standard)
  - [Polars TI DataFrame Extension](#polars-ti-dataframe-extension)
  - [Polars TI Strategy](#polars-ti-strategy)
- [Polars TI Strategies](#polars-ti-strategies)
  - [Types of Strategies](#types-of-strategies)
  - [Multiprocessing](#multiprocessing)
- [DataFrame Properties](#dataframe-properties)
- [DataFrame Methods](#dataframe-methods)
- [Indicators by Category](#indicators-by-category)
  - [Candles](#candles-64)
  - [Cycles](#cycles-2)
  - [Momentum](#momentum-43)
  - [Overlap](#overlap-36)
  - [Performance](#performance-3)
  - [Statistics](#statistics-10)
  - [Trend](#trend-24)
  - [Volatility](#volatility-16)
  - [Volume](#volume-20)
- [Performance Metrics](#performance-metrics)
- [Changes](#changes)
  - [General](#general)
  - [Breaking Indicators](#breaking-indicators)
  - [New Indicators](#new-indicators)
  - [Updated Indicators](#updated-indicators)
- [Sources](#sources)
- [Support](#support)
<!--te-->

<br/>

# **Features**

- A Free & Open Source library with a LARGE flat library structure similar to TA Lib.
  - [150+ indicators](#indicators-by-category) and utilities.
  - [60+ Candelstick Patterns](#candles-64) with [TA Lib](https://ta-lib.org/) installed.
- Performance improvements with [numba](https://github.com/numba/numba)
- A [Pandas DataFrame Extension](https://pandas.pydata.org/docs/development/extending.html) named "ti", that provides additional properties, methods, and indicators to simplify time series calculations of `ohlcv` columns.
- Indicator Equivalence
  - **Primarily:** TA Lib
  - **Secondarily:** TradingView
  - **Tertiary:** [Sierra Chart](https://search.sierrachart.com/?Query=indicators&submitted=true) | [MQL5](https://www.mql5.com) | [FM Labs](https://www.fmlabs.com/reference/default.htm) | [Pro Real Code](https://www.prorealcode.com/prorealtime-indicators) | [User 42](https://user42.tuxfamily.org/chart/manual/index.html) | [Technical Traders](http://technical.traders.com/tradersonline/FeedTT-2014.html) | etc
- And more ...

<br/>

# **Under Development**

**Polars TI** checks if the user has some common trading packages installed including but not limited to: [**TA Lib**](https://mrjbq7.github.io/ta-lib/), [**Vector BT**](https://github.com/polakowo/vectorbt), [**YFinance**](https://github.com/ranaroussi/yfinance) ... Much of which is _experimental_ and likely to break until it stabilizes more.

- If **TA Lib** installed, existing indicators will _eventually_ get a **TA Lib** version.
- Easy Downloading of _ohlcv_ data using [yfinance](https://github.com/ranaroussi/yfinance). See `help(ti.ticker)` and `help(ti.yf)` and examples below.
- Some Common Performance Metrics

<br/>

# Installation

We recommend using `uv` for the fastest and most reliable installation experience. Instructions for `pip` and `conda` are also provided.

### A Critical Prerequisite: TA-Lib

This library uses the `TA-Lib` library for many of its candlestick pattern calculations.

**Good news!** Starting with TA-Lib 0.6.5, **binary wheels are available** that include the underlying C library for most platforms (Linux x86_64/arm64, macOS x86_64/arm64, Windows x86_64/x86/arm64). For these platforms, you can simply install with:

```bash
pip install TA-Lib
```

<details>
<summary><b>Click if you need to build from source (rare)</b></summary>

If binary wheels are not available for your platform, you'll need to install the underlying TA-Lib C library first.

**Linux (Debian/Ubuntu)**

```bash
# Download and build from source
wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.8/ta-lib-0.6.8-src.tar.gz
tar -xzf ta-lib-0.6.8-src.tar.gz
cd ta-lib-0.6.8/
./configure --prefix=/usr
make
sudo make install
```

**Linux (Fedora/CentOS/RHEL)**

```bash
sudo dnf install ta-lib-devel
```

**macOS**

```bash
brew install ta-lib
```

**Windows**
Download and run the installer from [ta-lib-0.6.8-windows-x86_64.msi](https://github.com/ta-lib/ta-lib/releases/download/v0.6.8/ta-lib-0.6.8-windows-x86_64.msi).

</details>

---

### Method 1: `uv` (Recommended)

`uv` is an extremely fast Python package installer and resolver.

1.  **Install `uv`:**

    ```bash
    # macOS / Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Windows
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    # Create the venv
    uv venv

    # Activate it
    source .venv/bin/activate
    ```

3.  **Install `polars-ti`:**
    To get all features, including TA-Lib patterns and `yfinance` integration, install the `full` extra:
    ```bash
    uv pip install "polars-ti[full]"
    ```

### Method 2: `pip` and `venv`

1.  **Create and activate a virtual environment:**

    ```bash
    # Create the venv
    python -m venv .venv

    # Activate it (macOS/Linux)
    source .venv/bin/activate

    # Activate it (Windows)
    .venv\Scripts\activate
    ```

2.  **Install `polars-ti`:**
    ```bash
    pip install "polars-ti[full]"
    ```

### Method 3: `conda`

Using `conda` can simplify the installation of `TA-Lib`, as it handles the binary dependency automatically.

1.  **Create and activate a conda environment:**

    ```bash
    conda create -n polars-ti-env python=3.11
    conda activate polars-ti-env
    ```

2.  **Install TA-Lib and `polars-ti`:**
    Install `TA-Lib` from the `conda-forge` channel first, then install `polars-ti` using pip.
    ```bash
    conda install -c conda-forge ta-lib
    pip install "polars-ti[full]"
    ```

---

### Versions

#### Stable

The `pip` version is the last stable release.

```sh
$ uv pip install "polars-ti[full]"
```

#### Latest Version

Best choice!

- Includes all fixes and updates between **pypi** and what is covered in this README.

```sh
$ uv pip install "polars-ti[full] @ git+https://github.com/CMobley7/polars-ti"
```

#### Cutting Edge

This is the _Development Version_ which could have bugs and other undesireable side effects. Use at own risk!

```sh
$ uv pip install "polars-ti[full] @ git+https://github.com/CMobley7/polars-ti.git@development"
```

<br/>

# **Quick Start**

```python
import pandas as pd
import polars_ti as ti

df = pd.DataFrame() # Empty DataFrame

# Load data
df = pd.read_csv("path/to/symbol.csv", sep=",")
# OR if you have yfinance installed
df = df.ti.ticker("aapl")

# VWAP requires the DataFrame index to be a DatetimeIndex.
# Replace "datetime" with the appropriate column from your DataFrame
df.set_index(pd.DatetimeIndex(df["datetime"]), inplace=True)

# Calculate Returns and append to the df DataFrame
df.ti.log_return(cumulative=True, append=True)
df.ti.percent_return(cumulative=True, append=True)

# New Columns with results
df.columns

# Take a peek
df.tiil()

# vv Continue Post Processing vv
```

<br/>

# **Help**

**Some** indicator arguments have been reordered for consistency. Use `help(ti.indicator_name)` for more information or make a Pull Request to improve documentation.

```python
import pandas as pd
import polars_ti as ti

# Create a DataFrame so 'ti' can be used.
df = pd.DataFrame()

# Help about this, 'ti', extension
help(df.ti)

# List of all indicators
df.ti.indicators()

# Help about an indicator such as bbands
help(ti.bbands)
```

<br/>

# **Issues and Contributions**

Thanks for using **Polars TI**!
<br/>

- ### [Comments and Feedback](https://github.com/CMobley7/polars-ti/issues)
  - Have you read **_this_** document?
  - Are you running the latest version?
    - `$ pip install -U git+https://github.com/CMobley7/polars-ti`
  - Have you tried the [Examples](https://github.com/CMobley7/polars-ti/tree/main/examples/)?
    - Did they help?
    - What is missing?
    - Could you help improve them?
  - Did you know you can easily build _Custom Strategies_ with the **[Strategy](https://github.com/CMobley7/polars-ti/blob/main/examples/Polars_TI_Study_Examples.ipynb) Class**?
  - Documentation could _always_ be improved. Can you help contribute?

- ### [Bugs, Indicators or Feature Requests](https://github.com/CMobley7/polars-ti/issues)
  - First, search the _Closed_ Issues **before** you _Open_ a new Issue; it may have already been solved.
  - Please be as **detailed** as possible _with_ reproducible code, links if any, applicable screenshots, errors, logs, and data samples. You **will** be asked again if you provide nothing.
    - You want a new indicator not currently listed.
    - You want an alternate version of an existing indicator.
    - The indicator does not match another website, library, broker platform, language, et al.
      - Do you have correlation analysis to back your claim?
      - Can you contribute?
  - You **will** be asked to fill out an Issue even if you email my personally.

<br/>

# **Contributors**

_Thank you for your contributions!_

<a href="https://github.com/twopirllc"><img src="https://avatars.githubusercontent.com/u/18198392?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/AbyssAlora"><img src="https://avatars.githubusercontent.com/u/32155747?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/abmyii"><img src="https://avatars.githubusercontent.com/u/52673001?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/alexonab"><img src="https://avatars.githubusercontent.com/u/16689258?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/allahyarzadeh"><img src="https://avatars.githubusercontent.com/u/11909557?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/bizso09"><img src="https://avatars.githubusercontent.com/u/1904536?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/CMobley7"><img src="https://avatars.githubusercontent.com/u/10121829?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/codesutras"><img src="https://avatars.githubusercontent.com/u/56551122?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/DannyMartens"><img src="https://avatars.githubusercontent.com/u/37220423?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/DrPaprikaa"><img src="https://avatars.githubusercontent.com/u/64958936?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/daikts"><img src="https://avatars.githubusercontent.com/u/64799229?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/danlim-wz"><img src="https://avatars.githubusercontent.com/u/52344837?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/delicateear"><img src="https://avatars.githubusercontent.com/u/167213?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/dorren"><img src="https://avatars.githubusercontent.com/u/27552?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/edwardwang1"><img src="https://avatars.githubusercontent.com/u/15675816?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"></a> <a href="https://github.com/FGU1"><img src="https://avatars.githubusercontent.com/u/56175843?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/ffhirata"><img src="https://avatars.githubusercontent.com/u/44292530?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/floatinghotpot"><img src="https://avatars.githubusercontent.com/u/2339512?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/GSlinger"><img src="https://avatars.githubusercontent.com/u/24567123?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/JoeSchr"><img src="https://avatars.githubusercontent.com/u/8218910?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/lluissalord"><img src="https://avatars.githubusercontent.com/u/7021552?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/locupleto"><img src="https://avatars.githubusercontent.com/u/3994906?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/luisbarrancos"><img src="https://avatars.githubusercontent.com/u/387352?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/M6stafa"><img src="https://avatars.githubusercontent.com/u/7975309?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/maxdignan"><img src="https://avatars.githubusercontent.com/u/8184722?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/mchant"><img src="https://avatars.githubusercontent.com/u/8502845?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/mihakralj"><img src="https://avatars.githubusercontent.com/u/31756078?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/moritzgun"><img src="https://avatars.githubusercontent.com/u/68067719?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/NkosenhleDuma"><img src="https://avatars.githubusercontent.com/u/51145741?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/nicoloridulfo"><img src="https://avatars.githubusercontent.com/u/49532161?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/olafos"><img src="https://avatars.githubusercontent.com/u/2526551?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/pbrumblay"><img src="https://avatars.githubusercontent.com/u/2146159?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/RajeshDhalange"><img src="https://avatars.githubusercontent.com/u/32175897?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/rengel8"><img src="https://avatars.githubusercontent.com/u/34138513?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/rluong003"><img src="https://avatars.githubusercontent.com/u/42408939?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/SoftDevDanial"><img src="https://avatars.githubusercontent.com/u/64815604?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/schwaa"><img src="https://avatars.githubusercontent.com/u/2640598?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/tg12"><img src="https://avatars.githubusercontent.com/u/12201893?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/twrobel"><img src="https://avatars.githubusercontent.com/u/394724?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/WellMaybeItIs"><img src="https://avatars.githubusercontent.com/u/84646494?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/whubsch"><img src="https://avatars.githubusercontent.com/u/24275736?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/witokondoria"><img src="https://avatars.githubusercontent.com/u/5685669?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/wouldayajustlookatit"><img src="https://avatars.githubusercontent.com/u/44936661?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"></a> <a href="https://github.com/YuvalWein"><img src="https://avatars.githubusercontent.com/u/65113623?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a> <a href="https://github.com/zlpatel"><img src="https://avatars.githubusercontent.com/u/3293739?v=4" class="avatar-user" width="35px;" style="border-radius: 5px;"/></a>

<br/>

# **Programming Conventions**

**Polars TI** has three primary "styles" of processing Technical Indicators for your use case and/or requirements. They are: _Standard_, _DataFrame Extension_, and the _Polars TI Strategy_. Each with increasing levels of abstraction for ease of use. As you become more familiar with **Polars TI**, the simplicity and speed of using a _Polars TI Strategy_ may become more apparent. Furthermore, you can create your own indicators through Chaining or Composition. Lastly, each indicator either returns a _Series_ or a _DataFrame_ in Uppercase Underscore format regardless of style.

<br/>

# _Standard_

You explicitly define the input columns and take care of the output.

- `sma10 = ti.sma(df["Close"], length=10)`
  - Returns a Series with name: `SMA_10`
- `donchiandf = ti.donchian(df["HIGH"], df["low"], lower_length=10, upper_length=15)`
  - Returns a DataFrame named `DC_10_15` and column names: `DCL_10_15, DCM_10_15, DCU_10_15`
- `ema10_ohlc4 = ti.ema(ti.ohlc4(df["Open"], df["High"], df["Low"], df["Close"]), length=10)`
  - Chaining indicators is possible but you have to be explicit.
  - Since it returns a Series named `EMA_10`. If needed, you may need to uniquely name it.

<br/>

# _Polars TI DataFrame Extension_

Calling `df.ti` will automatically lowercase _OHLCVA_ to _ohlcva_: _open, high, low, close, volume_, _adj_close_. By default, `df.ti` will use the _ohlcva_ for the indicator arguments removing the need to specify input columns directly.

- `sma10 = df.ti.sma(length=10)`
  - Returns a Series with name: `SMA_10`
- `ema10_ohlc4 = df.ti.ema(close=df.ti.ohlc4(), length=10, suffix="OHLC4")`
  - Returns a Series with name: `EMA_10_OHLC4`
  - Chaining Indicators _require_ specifying the input like: `close=df.ti.ohlc4()`.
- `donchiandf = df.ti.donchian(lower_length=10, upper_length=15)`
  - Returns a DataFrame named `DC_10_15` and column names: `DCL_10_15, DCM_10_15, DCU_10_15`

Same as the last three examples, but appending the results directly to the DataFrame `df`.

- `df.ti.sma(length=10, append=True)`
  - Appends to `df` column name: `SMA_10`.
- `df.ti.ema(close=df.ti.ohlc4(append=True), length=10, suffix="OHLC4", append=True)`
  - Chaining Indicators _require_ specifying the input like: `close=df.ti.ohlc4()`.
- `df.ti.donchian(lower_length=10, upper_length=15, append=True)`
  - Appends to `df` with column names: `DCL_10_15, DCM_10_15, DCU_10_15`.

<br/>

# _Polars TI Strategy_

A **Polars TI** Strategy is a named group of indicators to be run by the _strategy_ method. All Strategies use **mulitprocessing** _except_ when using the `col_names` parameter (see [below](#multiprocessing)). There are different types of _Strategies_ listed in the following section.

<br/>

### Here are the previous _Styles_ implemented using a Strategy Class:

```python
# (1) Create the Strategy
MyStrategy = ti.Strategy(
    name="DCSMA10",
    ti=[
        {"kind": "ohlc4"},
        {"kind": "sma", "length": 10},
        {"kind": "donchian", "lower_length": 10, "upper_length": 15},
        {"kind": "ema", "close": "OHLC4", "length": 10, "suffix": "OHLC4"},
    ]
)

# (2) Run the Strategy
df.ti.strategy(MyStrategy, **kwargs)
```

<br/><br/>

# **Polars TI** _Strategies_

The _Strategy_ Class is a simple way to name and group your favorite technical indicators by using a _Data Class_. **Polars TI** comes with two prebuilt basic Strategies to help you get started: **AllStrategy** and **CommonStrategy**. A _Strategy_ can be as simple as the **CommonStrategy** or as complex as needed using Composition/Chaining.

- When using the _strategy_ method, **all** indicators will be automatically appended to the DataFrame `df`.
- You are using a Chained Strategy when you have the output of one indicator as input into one or more indicators in the same _Strategy_.
- **Note:** Use the 'prefix' and/or 'suffix' keywords to distinguish the composed indicator from it's default Series.

See the [Polars TI Strategy Examples Notebook](https://github.com/CMobley7/polars-ti/blob/main/examples/Polars_TI_Study_Examples.ipynb) for examples including _Indicator Composition/Chaining_.

## Strategy Requirements

- _name_: Some short memorable string. _Note_: Case-insensitive "All" is reserved.
- _ti_: A list of dicts containing keyword arguments to identify the indicator and the indicator's arguments
- **Note:** A Strategy will fail when consumed by Polars TI if there is no `{"kind": "indicator name"}` attribute. _Remember_ to check your spelling.

## Optional Parameters

- _description_: A more detailed description of what the Strategy tries to capture. Default: None
- _created_: At datetime string of when it was created. Default: Automatically generated.

<br/>

# Types of Strategies

## _Builtin_

```python
# Running the Builtin CommonStrategy as mentioned above
df.ti.strategy(ti.CommonStrategy)

# The Default Strategy is the ti.AllStrategy. The following are equivalent:
df.ti.strategy()
df.ti.strategy("All")
df.ti.strategy(ti.AllStrategy)
```

## _Categorical_

```python
# List of indicator categories
df.ti.categories

# Running a Categorical Strategy only requires the Category name
df.ti.strategy("Momentum") # Default values for all Momentum indicators
df.ti.strategy("overlap", length=42) # Override all Overlap 'length' attributes
```

## _Custom_

```python
# Create your own Custom Strategy
CustomStrategy = ti.Strategy(
    name="Momo and Volatility",
    description="SMA 50,200, BBANDS, RSI, MACD and Volume SMA 20",
    ti=[
        {"kind": "sma", "length": 50},
        {"kind": "sma", "length": 200},
        {"kind": "bbands", "length": 20},
        {"kind": "rsi"},
        {"kind": "macd", "fast": 8, "slow": 21},
        {"kind": "sma", "close": "volume", "length": 20, "prefix": "VOLUME"},
    ]
)
# To run your "Custom Strategy"
df.ti.strategy(CustomStrategy)
```

<br/>

# **Multiprocessing**

The **Polars TI** _strategy_ method utilizes **multiprocessing** for bulk indicator processing of all Strategy types with **ONE EXCEPTION!** When using the `col_names` parameter to rename resultant column(s), the indicators in `ti` array will be ran in order.

```python
# VWAP requires the DataFrame index to be a DatetimeIndex.
# * Replace "datetime" with the appropriate column from your DataFrame
df.set_index(pd.DatetimeIndex(df["datetime"]), inplace=True)

# Runs and appends all indicators to the current DataFrame by default
# The resultant DataFrame will be large.
df.ti.strategy()
# Or the string "all"
df.ti.strategy("all")
# Or the ti.AllStrategy
df.ti.strategy(ti.AllStrategy)

# Use verbose if you want to make sure it is running.
df.ti.strategy(verbose=True)

# Use timed if you want to see how long it takes to run.
df.ti.strategy(timed=True)

# Choose the number of cores to use. Default is all available cores.
# For no multiprocessing, set this value to 0.
df.ti.cores = 4

# Maybe you do not want certain indicators.
# Just exclude (a list of) them.
df.ti.strategy(exclude=["bop", "mom", "percent_return", "wcp", "pvi"], verbose=True)

# Perhaps you want to use different values for indicators.
# This will run ALL indicators that have fast or slow as parameters.
# Check your results and exclude as necessary.
df.ti.strategy(fast=10, slow=50, verbose=True)

# Sanity check. Make sure all the columns are there
df.columns
```

<br/>

## Custom Strategy without Multiprocessing

**Remember** These will not be utilizing **multiprocessing**

```python
NonMPStrategy = ti.Strategy(
    name="EMAs, BBs, and MACD",
    description="Non Multiprocessing Strategy by rename Columns",
    ti=[
        {"kind": "ema", "length": 8},
        {"kind": "ema", "length": 21},
        {"kind": "bbands", "length": 20, "col_names": ("BBL", "BBM", "BBU")},
        {"kind": "macd", "fast": 8, "slow": 21, "col_names": ("MACD", "MACD_H", "MACD_S")}
    ]
)
# Run it
df.ti.strategy(NonMPStrategy)
```

<br/><br/>

# **DataFrame Properties**

## **adjusted**

```python
# Set ta to default to an adjusted column, 'adj_close', overriding default 'close'.
df.ti.adjusted = "adj_close"
df.ti.sma(length=10, append=True)

# To reset back to 'close', set adjusted back to None.
df.ti.adjusted = None
```

## **categories**

```python
# List of Polars TI categories.
df.ti.categories
```

## **cores**

```python
# Set the number of cores to use for strategy multiprocessing
# Defaults to the number of cpus you have.
df.ti.cores = 4

# Set the number of cores to 0 for no multiprocessing.
df.ti.cores = 0

# Returns the number of cores you set or your default number of cpus.
df.ti.cores
```

## **datetime_ordered**

```python
# The 'datetime_ordered' property returns True if the DataFrame
# index is of Pandas datetime64 and df.index[0] < df.index[-1].
# Otherwise it returns False.
df.ti.datetime_ordered
```

## **exchange**

```python
# Sets the Exchange to use when calculating the last_run property. Default: "NYSE"
df.ti.exchange

# Set the Exchange to use.
# Available Exchanges: "ASX", "BMF", "DIFX", "FWB", "HKE", "JSE", "LSE", "NSE", "NYSE", "NZSX", "RTS", "SGX", "SSE", "TSE", "TSX"
df.ti.exchange = "LSE"
```

## **last_run**

```python
# Returns the time Polars TI was last run as a string.
df.ti.last_run
```

## **reverse**

```python
# The 'reverse' is a helper property that returns the DataFrame
# in reverse order.
df.ti.reverse
```

## **prefix & suffix**

```python
# Applying a prefix to the name of an indicator.
prehl2 = df.ti.hl2(prefix="pre")
print(prehl2.name)  # "pre_HL2"

# Applying a suffix to the name of an indicator.
endhl2 = df.ti.hl2(suffix="post")
print(endhl2.name)  # "HL2_post"

# Applying a prefix and suffix to the name of an indicator.
bothhl2 = df.ti.hl2(prefix="pre", suffix="post")
print(bothhl2.name)  # "pre_HL2_post"
```

## **time_range**

```python
# Returns the time range of the DataFrame as a float.
# By default, it returns the time in "years"
df.ti.time_range

# Available time_ranges include: "years", "months", "weeks", "days", "hours", "minutes". "seconds"
df.ti.time_range = "days"
df.ti.time_range # prints DataFrame time in "days" as float
```

## **to_utc**

```python
# Sets the DataFrame index to UTC format.
df.ti.to_utc
```

<br/><br/>

# **DataFrame Methods**

## **constants**

```python
import numpy as np

# Add constant '1' to the DataFrame
df.ti.constants(True, [1])
# Remove constant '1' to the DataFrame
df.ti.constants(False, [1])

# Adding constants for charting
import numpy as np
chart_lines = np.append(np.arange(-4, 5, 1), np.arange(-100, 110, 10))
df.ti.constants(True, chart_lines)
# Removing some constants from the DataFrame
df.ti.constants(False, np.array([-60, -40, 40, 60]))
```

## **indicators**

```python
# Prints the indicators and utility functions
df.ti.indicators()

# Returns a list of indicators and utility functions
ind_list = df.ti.indicators(as_list=True)

# Prints the indicators and utility functions that are not in the excluded list
df.ti.indicators(exclude=["cg", "pgo", "ui"])
# Returns a list of the indicators and utility functions that are not in the excluded list
smaller_list = df.ti.indicators(exclude=["cg", "pgo", "ui"], as_list=True)
```

## **ticker**

```python
# Download Chart history using yfinance. (pip install yfinance) https://github.com/ranaroussi/yfinance
# It uses the same keyword arguments as yfinance (excluding start and end)
df = df.ti.ticker("aapl") # Default ticker is "SPY"

# Period is used instead of start/end
# Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
# Default: "max"
df = df.ti.ticker("aapl", period="1y") # Gets this past year

# History by Interval by interval (including intraday if period < 60 days)
# Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
# Default: "1d"
df = df.ti.ticker("aapl", period="1y", interval="1wk") # Gets this past year in weeks
df = df.ti.ticker("aapl", period="1mo", interval="1h") # Gets this past month in hours

# BUT WAIT!! THERE'S MORE!!
help(ti.yf)
```

<br/><br/>

# **Indicators** (_by Category_)

### **Candles** (64)

Patterns that are **not bold**, require TA-Lib to be installed: `pip install TA-Lib`

- 2crows
- 3blackcrows
- 3inside
- 3linestrike
- 3outside
- 3starsinsouth
- 3whitesoldiers
- abandonedbaby
- advanceblock
- belthold
- breakaway
- closingmarubozu
- concealbabyswall
- counterattack
- darkcloudcover
- **doji**
- dojistar
- dragonflydoji
- engulfing
- eveningdojistar
- eveningstar
- gapsidesidewhite
- gravestonedoji
- hammer
- hangingman
- harami
- haramicross
- highwave
- hikkake
- hikkakemod
- homingpigeon
- identical3crows
- inneck
- **inside**
- invertedhammer
- kicking
- kickingbylength
- ladderbottom
- longleggeddoji
- longline
- marubozu
- matchinglow
- mathold
- morningdojistar
- morningstar
- onneck
- piercing
- rickshawman
- risefall3methods
- separatinglines
- shootingstar
- shortline
- spinningtop
- stalledpattern
- sticksandwich
- takuri
- tasukigap
- thrusting
- tristar
- unique3river
- upsidegap2crows
- xsidegap3methods
- _Heikin-Ashi_: **ha**
- _Z Score_: **cdl_z**

```python
# Get all candle patterns (This is the default behaviour)
df = df.ti.cdl_pattern(name="all")

# Get only one pattern
df = df.ti.cdl_pattern(name="doji")

# Get some patterns
df = df.ti.cdl_pattern(name=["doji", "inside"])
```

<br/>

### **Cycles** (2)

- _Even Better Sinewave_: **ebsw**
- _Reflex_: **reflex**

<br/>

### **Momentum** (43)

- _Awesome Oscillator_: **ao**
- _Absolute Price Oscillator_: **apo**
- _Bias_: **bias**
- _Balance of Power_: **bop**
- _BRAR_: **brar**
- _Commodity Channel Index_: **cci**
- _Chande Forecast Oscillator_: **cfo**
- _Center of Gravity_: **cg**
- _Chande Momentum Oscillator_: **cmo**
- _Coppock Curve_: **coppock**
- _Connors RSI (CRSI)_: **crsi**
- _Correlation Trend Indicator_: **cti**
  - A wrapper for `ti.linreg(series, r=True)`
- _Directional Movement_: **dm**
- _Efficiency Ratio_: **er**
- _Elder Ray Index_: **eri**
- _Fisher Transform_: **fisher**
- _Inertia_: **inertia**
- _KDJ_: **kdj**
- _KST Oscillator_: **kst**
- _Moving Average Convergence Divergence_: **macd**
- _Momentum_: **mom**
- _Pretty Good Oscillator_: **pgo**
- _Percentage Price Oscillator_: **ppo**
- _Psychological Line_: **psl**
- _Quantitative Qualitative Estimation_: **qqe**
- _Rate of Change_: **roc**
- _Relative Strength Index_: **rsi**
- _Relative Strength Xtra_: **rsx**
- _Relative Vigor Index_: **rvgi**
- _Slope_: **slope**
- _Smart Money Comcept_: **smc**
- _SMI Ergodic_ **smi**
- _Squeeze_: **squeeze**
  - Default is John Carter's. Enable Lazybear's with `lazybear=True`
- _Squeeze Pro_: **squeeze_pro**
- _Schaff Trend Cycle_: **stc**
- _Stochastic Oscillator_: **stoch**
- _Fast Stochastic_: **stochf**
- _Stochastic RSI_: **stochrsi**
- _True Momentum Oscillator_: **tmo**
- _Trix_: **trix**
- _True Strength Index_: **tsi**
- _Ultimate Oscillator_: **uo**
- _Williams %R_: **willr**

<br/>

### **Overlap** (36)

- _Bill Williams Alligator_: **alligator**
- _Arnaud Legoux Moving Average_: **alma**
- _Double Exponential Moving Average_: **dema**
- _Exponential Moving Average_: **ema**
- _Fibonacci's Weighted Moving Average_: **fwma**
- _Gann High-Low Activator_: **hilo**
- _High-Low Average_: **hl2**
- _High-Low-Close Average_: **hlc3**
  - Commonly known as 'Typical Price' in Technical Analysis literature
- _Hull Exponential Moving Average_: **hma**
- _Holt-Winter Moving Average_: **hwma**
- _Ichimoku Kinkō Hyō_: **ichimoku**
  - Returns two DataFrames. For more information: `help(ti.ichimoku)`.
  - `lookahead=False` drops the Chikou Span Column to prevent potential data leak.
- _Jurik Moving Average_: **jma**
- _Kaufman's Adaptive Moving Average_: **kama**
- _Linear Regression_: **linreg**
- _Ehler's MESA Adaptive Moving Average_: **mama**
- _McGinley Dynamic_: **mcgd**
- _Midpoint_: **midpoint**
- _Midprice_: **midprice**
- _Open-High-Low-Close Average_: **ohlc4**
- _Pivot Points_: **pivots**
- _Pascal's Weighted Moving Average_: **pwma**
- _WildeR's Moving Average_: **rma**
- _Sine Weighted Moving Average_: **sinwma**
- _Simple Moving Average_: **sma**
- _SMoothed Moving Average_: **ssma**
- _Ehler's Super Smoother Filter_: **ssf**
- _Ehler's 3 Pole Super Smoother Filter_: **ssf3**
- _Supertrend_: **supertrend**
- _Symmetric Weighted Moving Average_: **swma**
- _T3 Moving Average_: **t3**
- _Triple Exponential Moving Average_: **tema**
- _Triangular Moving Average_: **trima**
- _Variable Index Dynamic Average_: **vidya**
- _Weighted Closing Price_: **wcp**
- _Weighted Moving Average_: **wma**
- _Zero Lag Moving Average_: **zlma**

<br/>

### **Performance** (3)

Use parameter: cumulative=**True** for cumulative results.

- _Draw Down_: **drawdown**
- _Log Return_: **log_return**
- _Percent Return_: **percent_return**

<br/>

### **Statistics** (10)

- _Entropy_: **entropy**
- _Kurtosis_: **kurtosis**
- _Mean Absolute Deviation_: **mad**
- _Median_: **median**
- _Quantile_: **quantile**
- _Skew_: **skew**
- _Standard Deviation_: **stdev**
- _Think or Swim Standard Deviation All_: **tos_stdevall**
- _Variance_: **variance**
- _Z Score_: **zscore**

<br/>

### **Trend** (24)

- _Average Directional Movement Index_: **adx**
  - Also includes **dmp** and **dmn** in the resultant DataFrame.
- _Alpha Trend_: **alphatrend**
- _Archer Moving Averages Trends_: **amat**
- _Aroon & Aroon Oscillator_: **aroon**
- _Choppiness Index_: **chop**
- _Chande Kroll Stop_: **cksp**
- _Decay_: **decay**
- _Decreasing_: **decreasing**
- _Detrend Price Oscillator_: **dpo**
- _Hilbert Transform TrendLine_: **ht_trendline**
- _Detrended Price Oscillator_: **dpo**
  - Set `lookahead=False` to disable centering and remove potential data leak.
- _Increasing_: **increasing**
- _Long Run_: **long_run**
- _Parabolic Stop and Reverse_: **psar**
- _Q Stick_: **qstick**
- _Random Walk Index_: **rwi**
- _Short Run_: **short_run**
- _Trendflex_: **trendflex**
- _Trend Signals_: **tsignals**
- _TTM Trend_: **ttm_trend**
- _Vertical Horizontal Filter_: **vhf**
- _Vortex_: **vortex**
- _Cross Signals_: **xsignals**
- _Zigzag_: **zigzag**

<br/>

### **Volatility** (16)

- _Aberration_: **aberration**
- _Acceleration Bands_: **accbands**
- _Average True Range_: **atr**
- _ATR Trailing Stop_: **atrts**
- _Bollinger Bands_: **bbands**
- _Chandelier Exit_: **chandelier_exit**
- _Donchian Channel_: **donchian**
- _Holt-Winter Channel_: **hwc**
- _Keltner Channel_: **kc**
- _Mass Index_: **massi**
- _Normalized Average True Range_: **natr**
- _Price Distance_: **pdist**
- _Relative Volatility Index_: **rvi**
- _Elder's Thermometer_: **thermo**
- _True Range_: **true_range**
- _Ulcer Index_: **ui**

<br/>

### **Volume** (20)

- _Accumulation/Distribution Index_: **ad**
- _Accumulation/Distribution Oscillator_: **adosc**
- _Archer On-Balance Volume_: **aobv**
- _Chaikin Money Flow_: **cmf**
- _Elder's Force Index_: **efi**
- _Ease of Movement_: **eom**
- _Klinger Volume Oscillator_: **kvo**
- _Money Flow Index_: **mfi**
- _Negative Volume Index_: **nvi**
- _On-Balance Volume_: **obv**
- _Positive Volume Index_: **pvi**
- _Percentage Volume Oscillator_: **pvo**
- _Price-Volume_: **pvol**
- _Price Volume Rank_: **pvr**
- _Price Volume Trend_: **pvt**
- _Volume Heatmap_: **vhm**
- _Volume Profile_: **vp**
- _Volume Weighted Average Price_: **vwap**
  - Requires the DataFrame index to be a DatetimeIndex
- _Volume Weighted Moving Average_: **vwma**
- _Worden Brothers' Time Segmented Value_: **wb_tsv**

<br/><br/>

## Backtesting with **vectorbt**

For **easier** integration with **vectorbt**'s Portfolio `from_signals` method, the `ti.trend_return` method has been replaced with `ti.tsignals` method to simplify the generation of trading signals. For a comprehensive example, see the example Jupyter Notebook [VectorBT Backtest with Polars TI](https://github.com/CMobley7/polars-ti/blob/main/examples/VectorBT_Backtest_with_Polars_TI.ipynb) in the examples directory.

<br/>

### Brief example

- See the [**vectorbt**](https://polakowo.io/vectorbt/) website more options and examples.

```python
import pandas as pd
import polars_ti as ti
import vectorbt as vbt

df = pd.DataFrame().ti.ticker("AAPL") # requires 'yfinance' installed

# Create the "Golden Cross"
df["GC"] = df.ti.sma(50, append=True) > df.ti.sma(200, append=True)

# Create boolean Signals(TS_Entries, TS_Exits) for vectorbt
golden = df.ti.tsignals(df.GC, asbool=True, append=True)

# Sanity Check (Ensure data exists)
print(df)

# Create the Signals Portfolio
pf = vbt.Portfolio.from_signals(df.close, entries=golden.TS_Entries, exits=golden.TS_Exits, freq="D", init_cash=100_000, fees=0.0025, slippage=0.0025)

# Print Portfolio Stats and Return Stats
print(pf.stats())
print(pf.returns_stats())
```

<br/><br/>

# **Changes**

## **General**

<br />

## **Breaking / Depreciated Indicators**

<br/>

## **New Indicators**

<br/>

## **Updated Indicators**

<br />

# **Sources**

[Original TA-LIB](http://ta-lib.org/) | [TradingView](http://www.tradingview.com) | [Sierra Chart](https://search.sierrachart.com/?Query=indicators&submitted=true) | [MQL5](https://www.mql5.com) | [FM Labs](https://www.fmlabs.com/reference/default.htm) | [Pro Real Code](https://www.prorealcode.com/prorealtime-indicators) | [User 42](https://user42.tuxfamily.org/chart/manual/index.html)

<br/>
