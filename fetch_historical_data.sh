#!/bin/bash
#
# Script to fetch historical data from Kraken for model training
# This script will download OHLCV data for the past 2 years
# for various trading pairs and timeframes
#

# Configure API keys - defaults to environment variables
# but can be passed as command line arguments
KRAKEN_API_KEY=${KRAKEN_API_KEY:-""}
KRAKEN_API_SECRET=${KRAKEN_API_SECRET:-""}

# Default settings
TRADING_PAIRS=("SOLUSD" "BTCUSD" "ETHUSD")
TIMEFRAMES=("1m" "5m" "15m" "30m" "1h" "4h" "1d")
OUTPUT_DIR="historical_data"

# API endpoints
KRAKEN_API_URL="https://api.kraken.com"
KRAKEN_PUBLIC_ENDPOINT="/0/public/OHLC"

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Parse command line arguments
while [ "$1" != "" ]; do
    case $1 in
        --api-key )         shift
                            KRAKEN_API_KEY=$1
                            ;;
        --api-secret )      shift
                            KRAKEN_API_SECRET=$1
                            ;;
        --pairs )           shift
                            IFS=',' read -r -a TRADING_PAIRS <<< "$1"
                            ;;
        --timeframes )      shift
                            IFS=',' read -r -a TIMEFRAMES <<< "$1"
                            ;;
        --output-dir )      shift
                            OUTPUT_DIR=$1
                            ;;
        * )                 echo "Unknown parameter: $1"
                            exit 1
    esac
    shift
done

# Convert timeframe to Kraken interval (in minutes)
function get_interval() {
    local timeframe=$1
    case $timeframe in
        "1m" )  echo "1";;
        "5m" )  echo "5";;
        "15m" ) echo "15";;
        "30m" ) echo "30";;
        "1h" )  echo "60";;
        "4h" )  echo "240";;
        "1d" )  echo "1440";;
        * )     echo "60";; # default to 1h
    esac
}

# Get current timestamp
current_timestamp=$(date +%s)

# Calculate timestamp for 2 years ago (in seconds)
two_years_ago=$((current_timestamp - 63072000))

# Function to fetch historical data
function fetch_data() {
    local pair=$1
    local timeframe=$2
    local interval=$(get_interval $timeframe)
    local output_file="${OUTPUT_DIR}/${pair}_${timeframe}.csv"
    
    echo "Fetching historical data for ${pair} (${timeframe})..."
    
    # Construct URL with parameters
    local url="${KRAKEN_API_URL}${KRAKEN_PUBLIC_ENDPOINT}?pair=${pair}&interval=${interval}&since=${two_years_ago}"
    
    # Make API request
    response=$(curl -s "${url}")
    
    # Check for errors
    error=$(echo $response | jq -r '.error[]' 2>/dev/null)
    if [ ! -z "$error" ] && [ "$error" != "null" ]; then
        echo "Error fetching data for ${pair} (${timeframe}): ${error}"
        return 1
    fi
    
    # Extract results
    result=$(echo $response | jq -r ".result.${pair}")
    if [ "$result" == "null" ] || [ -z "$result" ]; then
        # Try alternative naming
        result=$(echo $response | jq -r '.result | to_entries[0].value')
    fi
    
    if [ "$result" == "null" ] || [ -z "$result" ]; then
        echo "No data returned for ${pair} (${timeframe})"
        return 1
    fi
    
    # Format as CSV (timestamp, open, high, low, close, vwap, volume, count)
    echo "timestamp,open,high,low,close,vwap,volume,count" > ${output_file}
    echo $result | jq -r '.[] | [.[0], .[1], .[2], .[3], .[4], .[5], .[6], .[7]] | @csv' >> ${output_file}
    
    # Count number of records
    record_count=$(wc -l < ${output_file})
    record_count=$((record_count - 1))  # Subtract header line
    
    echo "Downloaded ${record_count} records for ${pair} (${timeframe}). Saved to ${output_file}"
    return 0
}

# Print configuration
echo "========================================================"
echo "Fetching Historical Data from Kraken"
echo "Trading Pairs: ${TRADING_PAIRS[@]}"
echo "Timeframes: ${TIMEFRAMES[@]}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "========================================================"

# Fetch data for each pair and timeframe
for pair in "${TRADING_PAIRS[@]}"; do
    for timeframe in "${TIMEFRAMES[@]}"; do
        fetch_data "${pair}" "${timeframe}"
        # Add delay to avoid rate limiting
        sleep 1
    done
done

# Convert timestamp to ISO date format in CSV files
echo "========================================================"
echo "Converting timestamps to ISO date format..."

for file in ${OUTPUT_DIR}/*.csv; do
    echo "Processing ${file}..."
    # Create a temporary file
    tmp_file="${file}.tmp"
    
    # Copy header
    head -n 1 "${file}" > "${tmp_file}"
    
    # Process timestamp in data rows
    tail -n +2 "${file}" | while IFS=',' read timestamp rest; do
        # Convert Unix timestamp to ISO date format
        date=$(date -d "@$((timestamp/1000))" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || date -r "$((timestamp/1000))" "+%Y-%m-%d %H:%M:%S")
        echo "${date},${rest}" >> "${tmp_file}"
    done
    
    # Replace original file with temporary file
    mv "${tmp_file}" "${file}"
done

echo "========================================================"
echo "All historical data downloaded and processed successfully."
echo "========================================================"