echo "----- eth0 interface down"
ifconfig eth0 down

echo "----- Removing dependencies and modules"
rmmod dwmac_qcom_ethqos 
rmmod stmmac_platform 
rmmod stmmac 

echo "----- Insert New Module"
insmod stmmac.ko 

echo "----- Load Dependencies"
modprobe stmmac_platform 
modprobe dwmac_qcom_ethqos 

echo "----- Bring eth0 interface up"
dmesg | tail -20