if [ ! -e speedtest.json ]
then
    curl -s https://raw.githubusercontent.com/sivel/speedtest-cli/master/speedtest.py | python -  --json > speedtest.json
fi
export BANDWIDTH=`python -c "import json; speedtest = json.load(open('speedtest.json')); print(int(max(1, min(speedtest['upload'], speedtest['download']) / 1e6)))"`
echo "Internet Bandwidth (Mb/s) = $BANDWIDTH"

if [ ! -e p2p-keygen ]
then
    curl -L https://www.dropbox.com/s/p1hi93ahy5295jf/p2p-keygen?dl=1 > p2p-keygen
    chmod +x p2p-keygen
fi

if [ ! -e identity ]
then
    ./p2p-keygen -f ./identity
fi
