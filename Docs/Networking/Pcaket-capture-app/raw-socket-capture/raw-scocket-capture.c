// sudo apt install libpcap-dev
// gcc packet_count.c -o packet_count -lpcap
// sudo ./packet_count
/*
Output log:
roopak@roopak-Latitude-3440:~/repos/iq9/Docs/Networking/Pcaket-capture-app$ sudo ./packet_capture 
Capturing packets...
Packet #1 captured: Length = 60 bytes
Packet #2 captured: Length = 1292 bytes
Packet #3 captured: Length = 1292 bytes
Packet #4 captured: Length = 1292 bytes
Packet #5 captured: Length = 1292 bytes
Packet #6 captured: Length = 1292 bytes
Packet #7 captured: Length = 1292 bytes
Packet #8 captured: Length = 187 bytes
Packet #9 captured: Length = 535 bytes
Packet #10 captured: Length = 66 bytes
Packet #11 captured: Length = 794 bytes
Packet #12 captured: Length = 88 bytes
Packet #13 captured: Length = 60 bytes
Packet #14 captured: Length = 60 bytes
Packet #15 captured: Length = 121 bytes
Packet #16 captured: Length = 76 bytes
Packet #17 captured: Length = 60 bytes
Packet #18 captured: Length = 60 bytes
Packet #19 captured: Length = 60 bytes
Packet #20 captured: Length = 166 bytes
Packet #21 captured: Length = 167 bytes
Packet #22 captured: Length = 77 bytes
Packet #23 captured: Length = 75 bytes
Packet #24 captured: Length = 192 bytes
Packet #25 captured: Length = 65 bytes
Packet #26 captured: Length = 139 bytes
Packet #27 captured: Length = 85 bytes
Packet #28 captured: Length = 67 bytes
Packet #29 captured: Length = 60 bytes
Packet #30 captured: Length = 60 bytes
Packet #31 captured: Length = 60 bytes
Packet #32 captured: Length = 60 bytes
^C

Capture stopped.
Total packets received = 32

*/

#include <pcap.h>
#include <stdio.h>
#include <signal.h>

static unsigned long packet_count = 0;
pcap_t *handle;

/* Ctrl+C handler */
void stop_capture(int signo)
{
    printf("\n\nCapture stopped.\n");
    printf("Total packets received = %lu\n", packet_count);

    pcap_breakloop(handle);
}

void packet_handler(u_char *args,
                    const struct pcap_pkthdr *header,
                    const u_char *packet)
{
    packet_count++;

    printf("Packet #%lu captured: Length = %d bytes\n",
           packet_count,
           header->len);
}

int main()
{
    char errbuf[PCAP_ERRBUF_SIZE];

    /* Handle Ctrl+C */
    signal(SIGINT, stop_capture);

    /* Open interface for packet capture */
    handle = pcap_open_live(
                "wlp0s20f3",   // interface
                BUFSIZ,        // capture size
                1,             // promiscuous mode
                1000,          // timeout (ms)
                errbuf);

    if (handle == NULL)
    {
        printf("pcap_open_live failed: %s\n", errbuf);
        return -1;
    }

    printf("Capturing packets...\n");

    /* Capture packets continuously */
    pcap_loop(handle,
              0,
              packet_handler,
              NULL);

    pcap_close(handle);

    return 0;
}