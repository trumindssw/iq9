#!/usr/bin/env python3
"""
Step 1: Preprocess CIC-DDoS2019 Dataset - MEMORY EFFICIENT VERSION
Converts PCAP + CSV files to training-ready numpy format
Uses streaming and chunking to handle large CSVs
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
from scapy.all import rdpcap, IP, TCP, UDP
from tqdm import tqdm
from datetime import datetime
import hashlib

import config

class CICDDoS2019Preprocessor:
    """
    Memory-efficient preprocessor for CIC-DDoS2019 dataset
    """
    
    def __init__(self, pcap_dir, csv_dir, output_dir, verbose=True):
        self.pcap_dir = pcap_dir
        self.csv_dir = csv_dir
        self.output_dir = output_dir
        self.verbose = verbose
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Statistics
        self.stats = {
            'start_time': datetime.now().isoformat(),
            'total_packets': 0,
            'benign_packets': 0,
            'attack_packets': 0,
            'skipped_packets': 0,
            'csvs_loaded': 0,
            'attack_flows': 0,
            'pcaps_processed': 0
        }
        
        # Use hashed set for memory efficiency
        self.attack_flow_hashes = set()
    
    def log(self, message):
        """Print if verbose mode enabled"""
        if self.verbose:
            print(message)
    
    def hash_flow_tuple(self, flow_tuple):
        """
        Hash flow tuple to save memory
        Instead of storing strings, store hash integers
        """
        if flow_tuple is None:
            return None
        
        # Create string representation
        flow_str = f"{flow_tuple[0]}-{flow_tuple[1]}-{flow_tuple[2]}-{flow_tuple[3]}-{flow_tuple[4]}"
        
        # Hash to integer (much smaller than string)
        return int(hashlib.md5(flow_str.encode()).hexdigest()[:16], 16)
    
    def parse_flow_id(self, flow_id_str):
        """Parse Flow ID from CSV"""
        try:
            flow_id_str = str(flow_id_str).strip()
            parts = flow_id_str.split('-')
            
            if len(parts) < 5:
                return None
            
            src_ip = parts[0].strip()
            dst_ip = parts[1].strip()
            src_port = int(parts[2])
            dst_port = int(parts[3])
            protocol = int(parts[4])
            
            return (src_ip, dst_ip, src_port, dst_port, protocol)
            
        except (ValueError, IndexError, AttributeError):
            return None
    
    def load_single_csv_chunked(self, csv_path, chunk_size=50000):
        """
        Load CSV in chunks to avoid memory overflow
        Process and hash flows incrementally
        """
        attack_name = os.path.basename(csv_path).replace('.csv', '')
        flow_count = 0
        
        try:
            # Get total rows for progress bar
            total_rows = sum(1 for _ in open(csv_path)) - 1  # -1 for header
            
            self.log(f"   Processing {attack_name} ({total_rows:,} rows)...")
            
            # Read CSV in chunks
            chunk_iterator = pd.read_csv(
                csv_path, 
                chunksize=chunk_size,
                low_memory=False
            )
            
            for chunk in tqdm(chunk_iterator, 
                            total=(total_rows // chunk_size) + 1,
                            desc=f"   {attack_name}",
                            disable=not self.verbose):
                
                # Find Flow ID column
                flow_id_col = None
                for col_name in chunk.columns:
                    if 'flow' in col_name.lower() and 'id' in col_name.lower():
                        flow_id_col = col_name
                        break
                
                if flow_id_col is None:
                    continue
                
                # Process flows in this chunk
                for flow_id in chunk[flow_id_col]:
                    flow_tuple = self.parse_flow_id(flow_id)
                    if flow_tuple:
                        flow_hash = self.hash_flow_tuple(flow_tuple)
                        if flow_hash:
                            self.attack_flow_hashes.add(flow_hash)
                            flow_count += 1
                
                # Explicitly delete chunk to free memory
                del chunk
            
            self.log(f"   ✅ {attack_name}: {flow_count:,} flows")
            
        except Exception as e:
            self.log(f"   ❌ Error loading {attack_name}: {e}")
        
        return flow_count
    
    def load_all_attack_flows(self):
        """Load all attack flows using chunked reading"""
        self.log("="*70)
        self.log("📂 STEP 1: Loading Attack Flows from CSV Files")
        self.log("="*70)
        
        csv_files = glob.glob(os.path.join(self.csv_dir, "*.csv"))
        
        if len(csv_files) == 0:
            self.log(f"❌ No CSV files found in {self.csv_dir}")
            return
        
        self.log(f"Found {len(csv_files)} CSV files\n")
        
        total_flows = 0
        
        for csv_file in sorted(csv_files):
            flows = self.load_single_csv_chunked(csv_file)
            total_flows += flows
            self.stats['csvs_loaded'] += 1
        
        self.stats['attack_flows'] = len(self.attack_flow_hashes)
        
        self.log(f"\n{'='*70}")
        self.log(f"✅ Total Attack Flows Loaded: {len(self.attack_flow_hashes):,}")
        self.log(f"   Memory usage: ~{len(self.attack_flow_hashes) * 16 / (1024**2):.1f} MB")
        self.log(f"{'='*70}\n")
    
    def extract_flow_tuple(self, packet):
        """Extract 5-tuple from packet"""
        if not packet.haslayer(IP):
            return None
        
        ip_layer = packet[IP]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        protocol = ip_layer.proto
        
        if packet.haslayer(TCP):
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
        elif packet.haslayer(UDP):
            src_port = packet[UDP].sport
            dst_port = packet[UDP].dport
        else:
            src_port = 0
            dst_port = 0
        
        return (src_ip, dst_ip, src_port, dst_port, protocol)
    
    def label_packet(self, packet):
        """Determine if packet is Attack or Benign using hashed lookup"""
        flow_tuple = self.extract_flow_tuple(packet)
        
        if flow_tuple is None:
            return None
        
        # Check forward direction
        flow_hash = self.hash_flow_tuple(flow_tuple)
        if flow_hash in self.attack_flow_hashes:
            return 1
        
        # Check reverse direction
        reverse_tuple = (flow_tuple[1], flow_tuple[0], 
                        flow_tuple[3], flow_tuple[2], 
                        flow_tuple[4])
        
        reverse_hash = self.hash_flow_tuple(reverse_tuple)
        if reverse_hash in self.attack_flow_hashes:
            return 1
        
        return 0
    
    def mask_ip_addresses(self, ip_packet_bytes):
        """Mask source and destination IP addresses"""
        packet_array = bytearray(ip_packet_bytes)
        
        if len(packet_array) >= 20:
            packet_array[12:16] = b'\x00\x00\x00\x00'
            packet_array[16:20] = b'\x00\x00\x00\x00'
        
        return bytes(packet_array)
    
    def preprocess_packet(self, packet):
        """Convert packet to 1500-byte array"""
        if not packet.haslayer(IP):
            return None
        
        ip_packet = bytes(packet[IP])
        
        if config.MASK_IPS:
            ip_packet = self.mask_ip_addresses(ip_packet)
        
        if len(ip_packet) < config.PACKET_SIZE:
            padded = ip_packet + b'\x00' * (config.PACKET_SIZE - len(ip_packet))
        else:
            padded = ip_packet[:config.PACKET_SIZE]
        
        return np.frombuffer(padded, dtype=np.uint8)
    
    def process_single_pcap(self, pcap_path):
        """Process one PCAP file"""
        filename = os.path.basename(pcap_path)
        self.log(f"\n📦 Processing: {filename}")
        
        X_batch = []
        y_batch = []
        
        try:
            packets = rdpcap(pcap_path)
            self.log(f"   Total packets: {len(packets):,}")
            
            benign_count = 0
            attack_count = 0
            skipped_count = 0
            
            for packet in tqdm(packets, desc=f"   {filename}", 
                             disable=not self.verbose):
                
                processed = self.preprocess_packet(packet)
                if processed is None:
                    skipped_count += 1
                    continue
                
                label = self.label_packet(packet)
                if label is None:
                    skipped_count += 1
                    continue
                
                X_batch.append(processed)
                y_batch.append(label)
                
                if label == 0:
                    benign_count += 1
                else:
                    attack_count += 1
            
            self.stats['total_packets'] += len(packets)
            self.stats['benign_packets'] += benign_count
            self.stats['attack_packets'] += attack_count
            self.stats['skipped_packets'] += skipped_count
            self.stats['pcaps_processed'] += 1
            
            total_labeled = benign_count + attack_count
            if total_labeled > 0:
                self.log(f"   ✅ Labeled: {total_labeled:,} packets")
                self.log(f"      Benign: {benign_count:,} ({benign_count/total_labeled*100:.1f}%)")
                self.log(f"      Attack: {attack_count:,} ({attack_count/total_labeled*100:.1f}%)")
                self.log(f"      Skipped: {skipped_count:,}")
            
        except Exception as e:
            self.log(f"   ❌ Error: {e}")
        
        return X_batch, y_batch
    
    def process_all_pcaps(self, pcap_limit=None):
        self.log("="*70)
        self.log("📦 STEP 2: Processing PCAP Files")
        self.log("="*70)
    
    # Look for PCAP files with multiple patterns
        pcap_files = []
        
        # Pattern 1: Standard .pcap extension
        pattern1 = os.path.join(self.pcap_dir, "*.pcap")
        pcap_files.extend(glob.glob(pattern1))
        
        # Pattern 2: CIC-DDoS2019 format (no extension)
        pattern2 = os.path.join(self.pcap_dir, "SAT-*")
        pcap_files.extend(glob.glob(pattern2))
        
        # Remove duplicates and sort
        pcap_files = sorted(list(set(pcap_files)))
    
        if len(pcap_files) == 0:
            self.log(f"❌ No PCAP files found in {self.pcap_dir}")
            self.log(f"   Tried: {pattern1}")
            self.log(f"   Tried: {pattern2}")
            return None, None
    
        if pcap_limit:
            pcap_files = pcap_files[:pcap_limit]
        
        self.log(f"Found {len(pcap_files)} PCAP files to process\n")
    
        all_X = []
        all_y = []
    
        for pcap_file in pcap_files:
            X_batch, y_batch = self.process_single_pcap(pcap_file)
            all_X.extend(X_batch)
            all_y.extend(y_batch)
    
        if len(all_X) == 0:
            self.log("❌ No packets were successfully processed")
            return None, None
        
        X = np.array(all_X, dtype=np.uint8)
        y = np.array(all_y, dtype=np.int32)
    
        return X, y

    def save_dataset(self, X, y, filename):
        """Save preprocessed dataset"""
        self.log("\n" + "="*70)
        self.log("💾 STEP 3: Saving Dataset")
        self.log("="*70)
        
        output_path = os.path.join(self.output_dir, filename)
        
        np.savez_compressed(
            output_path,
            X=X,
            y=y,
            packet_size=config.PACKET_SIZE,
            stats=self.stats
        )
        
        file_size_gb = os.path.getsize(output_path) / (1024**3)
        self.log(f"\n✅ Saved to: {output_path}")
        self.log(f"   File size: {file_size_gb:.2f} GB")
        
        return output_path
    
    def save_stats(self):
        """Save statistics to JSON"""
        self.stats['end_time'] = datetime.now().isoformat()
        stats_path = os.path.join(config.LOGS_DIR, 'preprocessing_stats.json')
        
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        self.log(f"   Stats saved to: {stats_path}")
    
    def print_final_stats(self, X, y):
        """Print final dataset statistics"""
        self.log("\n" + "="*70)
        self.log("📊 FINAL STATISTICS")
        self.log("="*70)
        
        self.log(f"\n🔢 Dataset Shape:")
        self.log(f"   X: {X.shape}")
        self.log(f"   y: {y.shape}")
        
        self.log(f"\n📈 Label Distribution:")
        self.log(f"   Total packets: {len(y):,}")
        self.log(f"   Benign (0): {np.sum(y==0):,} ({np.sum(y==0)/len(y)*100:.1f}%)")
        self.log(f"   Attack (1): {np.sum(y==1):,} ({np.sum(y==1)/len(y)*100:.1f}%)")
        
        self.log(f"\n📁 CSV Processing:")
        self.log(f"   CSVs loaded: {self.stats['csvs_loaded']}")
        self.log(f"   Attack flows: {self.stats['attack_flows']:,}")
        
        self.log(f"\n📦 PCAP Processing:")
        self.log(f"   PCAPs processed: {self.stats['pcaps_processed']}")
        self.log(f"   Total packets read: {self.stats['total_packets']:,}")
        self.log(f"   Packets labeled: {len(y):,}")
        self.log(f"   Packets skipped: {self.stats['skipped_packets']:,}")
        
        self.log("="*70 + "\n")
    
    def run(self):
        """Run complete preprocessing pipeline"""
        self.log("\n" + "🚀" + "="*68 + "🚀")
        self.log("   CIC-DDoS2019 Preprocessor - Memory Efficient")
        self.log("🚀" + "="*68 + "🚀" + "\n")
        
        # Step 1: Load attack flows (chunked)
        self.load_all_attack_flows()
        
        if len(self.attack_flow_hashes) == 0:
            self.log("❌ No attack flows loaded. Exiting.")
            return None, None
        
        # Step 2: Process PCAPs
        X, y = self.process_all_pcaps(config.PCAP_LIMIT)
        
        if X is None or len(X) == 0:
            self.log("❌ No packets processed. Exiting.")
            return None, None
        
        # Step 3: Save dataset
        output_path = self.save_dataset(X, y, config.FULL_DATASET_NAME)
        
        # Step 4: Save stats
        self.save_stats()
        
        # Step 5: Print statistics
        self.print_final_stats(X, y)
        
        return X, y


def main():
    print("Starting CIC-DDoS2019 preprocessing (Memory Efficient Version)...")
    print(f"PCAP directory: {config.PCAP_DIR}")
    print(f"CSV directory: {config.CSV_DIR}")
    print(f"Output directory: {config.PREPROCESSED_DIR}")
    
    if config.PCAP_LIMIT:
        print(f"⚠️  WARNING: Processing only first {config.PCAP_LIMIT} PCAPs (test mode)")
    
    input("\nPress Enter to continue...")
    
    preprocessor = CICDDoS2019Preprocessor(
        pcap_dir=config.PCAP_DIR,
        csv_dir=config.CSV_DIR,
        output_dir=config.PREPROCESSED_DIR,
        verbose=True
    )
    
    X, y = preprocessor.run()
    
    if X is not None:
        print("\n✅ Step 1 complete! Next run: python 2_split_dataset.py")
    else:
        print("\n❌ Preprocessing failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
