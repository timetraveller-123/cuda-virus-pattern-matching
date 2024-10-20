#include "kseq/kseq.h"
#include "common.h"
#include <iostream>

__global__ void matchSequences(const char* d_samples, const char* d_signatures, 
                               const int numSignatures, const int sampleLength, 
                               const int signatureLength, int* d_matches, const char* d_samplescores) {

    int sampleIdx = blockIdx.x;        // Block index corresponds to the sample index
    int signatureIdx = threadIdx.x;    // Thread index corresponds to the signature index
    
    if(signatureIdx < numSignatures){
        const char* sampleSeq = d_samples + sampleIdx * sampleLength;
        
        const char* signatureSeq = d_signatures + signatureIdx * signatureLength;
    
        bool matched = false;

        // Compare signature against the sample (handling 'N' as wildcard and skipping 'X')
        for (int i = 0; i <= sampleLength - signatureLength; i++) {
            matched = true;
            int score = 0;
            for (int j = 0; j < signatureLength; j++) {
                
                // if (signatureSeq[j] != 'X' && sampleSeq[i + j] == 'X') {
                //     // Skip comparison if padding is encountered
                //     matched = false;
                //     break;
                // }

                // if(signatureSeq[j] == 'X') {
                //     continue;
                // }

                // if (signatureSeq[j] != 'N' && sampleSeq[i + j] != 'N' && signatureSeq[j] != sampleSeq[i + j]) {
                //     matched = false;
                //     break;
                // }
                // score += ((int)d_samplescores[sampleIdx*sampleLength +i+j] - 33);

                //char sa = sampleSeq[i + j], si = signatureSeq[j];
                char sa = sampleSeq[i + j], si = d_signatures[j*numSignatures + signatureIdx];

                bool same = sa == si, san = sa == 'N', sin = si == 'N';

                score += ((int)d_samplescores[sampleIdx*sampleLength + i + j]-33)*(same || sin);
                matched = matched & (si == 'X' || (sa != 'X' && (same || san || sin)));
                if(matched == false)break;
            }
            __syncwarp();
            if (matched) {
                // Mark this sample-signature pair as matched
                d_matches[sampleIdx * numSignatures + signatureIdx] = score;
                return;
            }
        }
    }

}

void runMatcher(const std::vector<klibpp::KSeq>& samples, const std::vector<klibpp::KSeq>& signatures, std::vector<MatchResult>& matches) {

    int numSamples = samples.size();
    int numSignatures = signatures.size();
    const int fixedSampleLength = 200000;      // Fixed length for samples
    const int fixedSignatureLength = 10016;   // Fixed length for signatures


    // Pad sequences to the fixed length
    std::vector<char> h_sampleData(numSamples * fixedSampleLength, 'X');  // Pad with 'X'
    std::vector<char> h_signatureData(numSignatures * fixedSignatureLength, 'X');
    std::vector<char> h_samplescores(numSamples * fixedSampleLength, 33);  // Initialize with default 0s

    for (int i = 0; i < numSamples; i++) {
        memcpy(h_sampleData.data() + i * fixedSampleLength, samples[i].seq.c_str(), samples[i].seq.size());
    }

    for (int i = 0; i < numSignatures; i++) {
        //memcpy(h_signatureData.data() + i * fixedSignatureLength, signatures[i].seq.c_str(), signatures[i].seq.size());
        for(int j = 0; j < signatures[i].seq.size(); j++){
            h_signatureData[j*numSignatures + i] = signatures[i].seq[j];
        }
    }

    
    // Fill associated numbers for each sample
    for (int i = 0; i < numSamples; i++) {
        memcpy(h_samplescores.data() + i * fixedSampleLength, samples[i].qual.c_str(), samples[i].qual.size());
    }


    // Allocate device memory
    char* d_samples;
    char* d_signatures;
    int* d_matches;
    char* d_sample_scores;

    cudaMalloc(&d_samples, numSamples * fixedSampleLength * sizeof(char));
    cudaMalloc(&d_signatures, numSignatures * fixedSignatureLength * sizeof(char));
    cudaMalloc(&d_matches, numSamples * numSignatures * sizeof(int));
    cudaMalloc(&d_sample_scores, numSamples * fixedSampleLength * sizeof(char));

    auto start_wall = std::chrono::high_resolution_clock::now();
    // Copy data to device
    cudaMemcpy(d_samples, h_sampleData.data(), numSamples * fixedSampleLength * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_signatures, h_signatureData.data(), numSignatures * fixedSignatureLength * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemset(d_matches, -1, numSamples * numSignatures * sizeof(int));
    cudaMemcpy(d_sample_scores, h_samplescores.data(), numSamples * fixedSampleLength * sizeof(char), cudaMemcpyHostToDevice);

    auto end_wall = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> wall_elapsed_seconds = end_wall - start_wall;

    std::cerr << "(FOR AUTOMATED CHECKING) Total runMatcher time:" << wall_elapsed_seconds.count() << "s" << std::endl;


    // Launch the kernel
    int threadsPerBlock = 1024;  // Number of threads per block is the number of signatures
    int numBlocks = numSamples;           // Number of blocks is the number of samples

    matchSequences<<<numBlocks, threadsPerBlock>>>(d_samples, d_signatures, 
                                                   numSignatures, 
                                                   fixedSampleLength, fixedSignatureLength, 
                                                   d_matches, d_sample_scores);
    cudaDeviceSynchronize();


    

    // Copy matches back to host
    std::vector<int> h_matches(numSamples * numSignatures);
    cudaMemcpy(h_matches.data(), d_matches, numSamples * numSignatures * sizeof(int), cudaMemcpyDeviceToHost);

    
    // Gather results
    for (int i = 0; i < numSamples; i++) {
        for (int j = 0; j < numSignatures; j++) {
            if (h_matches[i * numSignatures + j] >0) {
                MatchResult result;
                result.sample_name = samples[i].name;
                result.signature_name = signatures[j].name;
                result.match_score = (float)h_matches[i*numSignatures + j]/signatures[j].seq.size();
                matches.push_back(result);
            }
        }
    }



    // Free device memory
    cudaFree(d_samples);
    cudaFree(d_signatures);
    cudaFree(d_matches);
    cudaFree(d_sample_scores);

}
