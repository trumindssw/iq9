// compile - gcc packet_sniffer.c -o packet_sniffer

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <linux/if_packet.h>
#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <netinet/udp.h>
#include <net/ethernet.h>

#define BUFFER_SIZE 65536

void inspect_packet(unsigned char *buffer, int size)
{
   struct ethhdr *eth = (struct ethhdr *)buffer;

   printf("\n==== New Packet ====\n");

   // L2 - Ethernet
   printf("L2 Ethernet Header\n");
   printf("Source MAC: %02x:%02x:%02x:%02x:%02x:%02x\n",
          eth->h_source[0], eth->h_source[1], eth->h_source[2],
          eth->h_source[3], eth->h_source[4], eth->h_source[5]);

   printf("Destination MAC: %02x:%02x:%02x:%02x:%02x:%02x\n",
          eth->h_dest[0], eth->h_dest[1], eth->h_dest[2],
          eth->h_dest[3], eth->h_dest[4], eth->h_dest[5]);

   // Check if IP
   if (ntohs(eth->h_proto) == ETH_P_IP)
   {
       struct iphdr *ip = (struct iphdr *)(buffer + sizeof(struct ethhdr));

       struct sockaddr_in src, dest;
       src.sin_addr.s_addr = ip->saddr;
       dest.sin_addr.s_addr = ip->daddr;

       printf("\nL3 IP Header\n");
       printf("Source IP: %s\n", inet_ntoa(src.sin_addr));
       printf("Destination IP: %s\n", inet_ntoa(dest.sin_addr));
       printf("Protocol: %d\n", ip->protocol);

       // L4
       if (ip->protocol == IPPROTO_TCP)
       {
           struct tcphdr *tcp = (struct tcphdr *)(buffer +
                                   sizeof(struct ethhdr) +
                                   ip->ihl * 4);

           printf("\nL4 TCP Header\n");
           printf("Source Port: %u\n", ntohs(tcp->source));
           printf("Destination Port: %u\n", ntohs(tcp->dest));
           printf("SYN: %d ACK: %d\n", tcp->syn, tcp->ack);

           unsigned char *payload = buffer +
               sizeof(struct ethhdr) +
               ip->ihl * 4 +
               tcp->doff * 4;

           int payload_size = size -
               (sizeof(struct ethhdr) +
                ip->ihl * 4 +
                tcp->doff * 4);

           if (payload_size > 0)
           {
               printf("\nL7 Payload (first 20 bytes):\n");
               for (int i = 0; i < payload_size && i < 20; i++)
                   printf("%02x ", payload[i]);
               printf("\n");
           }
       }
       else if (ip->protocol == IPPROTO_UDP)
       {
           struct udphdr *udp = (struct udphdr *)(buffer +
                                   sizeof(struct ethhdr) +
                                   ip->ihl * 4);

           printf("\nL4 UDP Header\n");
           printf("Source Port: %u\n", ntohs(udp->source));
           printf("Destination Port: %u\n", ntohs(udp->dest));
       }
   }
}

int main()
{
   int sock_raw;
   unsigned char *buffer = (unsigned char *)malloc(BUFFER_SIZE);

   sock_raw = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
   if (sock_raw < 0)
   {
       perror("Socket Error");
       return 1;
   }

   printf("Listening for packets...\n");

   while (1)
   {
       int data_size = recvfrom(sock_raw, buffer, BUFFER_SIZE, 0, NULL, NULL);
       if (data_size < 0)
       {
           perror("Recv Error");
           return 1;
       }

       inspect_packet(buffer, data_size);
   }

   close(sock_raw);
   free(buffer);
   return 0;
}
