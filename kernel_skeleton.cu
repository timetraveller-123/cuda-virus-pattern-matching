#include "kseq/kseq.h"
#include "common.h"
#include <iostream>

__device__ __constant__ const int fixedSignatureLength = 10240;
__device__ __constant__ const int fixedSampleLength = 200000;

__device__ __constant__ const int samplesPerBlock = 8, signaturesPerBlock = 4;


__global__ void matchSequences(const char* d_samples, const char* d_signatures, 
                               const int numSignatures, const int numSamples, int* d_matches, const char* d_samplescores) {



    int sampleIdx = blockIdx.x * blockDim.x + threadIdx.x;        
    int signatureIdx = blockIdx.y * blockDim.y + threadIdx.y;    

    __shared__ char signature[signaturesPerBlock*fixedSignatureLength];
    if(threadIdx.x == 0 && signatureIdx < numSignatures){
        for(int j = 0; j < 10240; j++){
            signature[j*blockDim.y + threadIdx.y] = d_signatures[signatureIdx*fixedSignatureLength + j];
        }
    }
    
    __syncthreads();
    if(signatureIdx < numSignatures && sampleIdx < numSamples){
    
        bool matched = false;

        for (int i = 0; i <= fixedSampleLength - fixedSignatureLength; i++) {
            matched = true;
            int score = 0;
            for (int j = 0; j < fixedSignatureLength; j++) {

                char sa = d_samples[sampleIdx*fixedSampleLength + i + j];
                char si =  signature[j*blockDim.y + threadIdx.y] ;   //  d_signatures[signatureIdx*fixedSignatureLength+j]; //

                bool same = sa == si, san = sa == 'N', sin = si == 'N';

                score += ((int)d_samplescores[sampleIdx*fixedSampleLength + i + j]-33)*(same || sin);
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



    // Pad sequences to the fixed length
    std::vector<char> h_sampleData(numSamples * fixedSampleLength, 'X');  // Pad with 'X'
    std::vector<char> h_signatureData(numSignatures * fixedSignatureLength, 'X');
    std::vector<char> h_samplescores(numSamples * fixedSampleLength, 33);  // Initialize with default 0s

    for (int i = 0; i < numSamples; i++) {
        memcpy(h_sampleData.data() + i * fixedSampleLength, samples[i].seq.c_str(), samples[i].seq.size());
    }

    for (int i = 0; i < numSignatures; i++) {
        memcpy(h_signatureData.data() + i * fixedSignatureLength, signatures[i].seq.c_str(), signatures[i].seq.size());
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


    
    
    dim3 gridSize(2200/samplesPerBlock + 1, 1024/signaturesPerBlock + 1);
    dim3 blockSize(samplesPerBlock, signaturesPerBlock);
    matchSequences<<<gridSize, blockSize>>>(d_samples, d_signatures, 
                                                   numSignatures, numSamples,
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
