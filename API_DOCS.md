# RAG Backend API Documentation

## Overview

This document provides comprehensive API documentation for the RAG (Retrieval-Augmented Generation) Backend API, designed for integration with Android applications.

## Base URL

### Development/Testing
- **Local Machine**: `http://localhost:8000`
- **Android Emulator**: `http://10.0.2.2:8000`
  - **Important**: When testing from an Android emulator, use `10.0.2.2` instead of `localhost` or `127.0.0.1`. This is the special IP address that the Android emulator uses to refer to the host machine's localhost.

### Production
- Replace with your production server URL (e.g., `https://api.yourdomain.com`)

## API Endpoints

### 1. Health Check

**Endpoint**: `GET /health`

**Description**: Check if the server is running and the vector store is loaded.

**Response**:
```json
{
  "status": "healthy",
  "vectorstore_loaded": true,
  "embeddings_loaded": true
}
```

### 2. Augment Query (Main Endpoint)

**Endpoint**: `POST /api/v1/augment`

**Description**: Augment a user query with relevant context from the vector store. This is the primary endpoint for RAG functionality.

#### Request

**Method**: `POST`

**Path**: `/api/v1/augment`

**Headers**:
```
Content-Type: application/json
```

**Request Body** (JSON):
```json
{
  "query": "What is the main topic of the document?",
  "k": 3
}
```

**Fields**:
- `query` (string, required): The user's question or query string
- `k` (integer, optional): Number of documents to retrieve. Default: 3, Range: 1-20

**Example Request**:
```json
{
  "query": "How do coroutines work in Kotlin?",
  "k": 5
}
```

#### Response

**Success Response** (200 OK):
```json
{
  "original_query": "How do coroutines work in Kotlin?",
  "context_chunks": [
    "Coroutines are a concurrency design pattern that you can use on Android to simplify code that executes asynchronously. Coroutines were added to Kotlin in version 1.3...",
    "A coroutine is an instance of suspendable computation. It is conceptually similar to a thread, in that it takes a block of code to run that works concurrently with the rest of the code...",
    "Coroutines follow a principle of structured concurrency which means that new coroutines can only be launched in a specific CoroutineScope..."
  ],
  "suggested_prompt": "Context:\n[Context 1]: Coroutines are a concurrency design pattern that you can use on Android to simplify code that executes asynchronously. Coroutines were added to Kotlin in version 1.3...\n\n[Context 2]: A coroutine is an instance of suspendable computation. It is conceptually similar to a thread, in that it takes a block of code to run that works concurrently with the rest of the code...\n\n[Context 3]: Coroutines follow a principle of structured concurrency which means that new coroutines can only be launched in a specific CoroutineScope...\n\nQuestion: How do coroutines work in Kotlin?\n\nPlease answer the question based on the provided context."
}
```

**Error Responses**:

- **503 Service Unavailable**: Vector store not loaded
```json
{
  "detail": "Vector store or embeddings not loaded. Please check server logs."
}
```

- **500 Internal Server Error**: Server error during processing
```json
{
  "detail": "Internal server error: [error message]"
}
```

## Integration Logic for Android App

### Step-by-Step Integration Guide

1. **User Input**: User enters a question in your Android app

2. **Call RAG API**: Send a POST request to `/api/v1/augment` with the user's query
   ```kotlin
   // Example using Retrofit/OkHttp
   val request = AugmentRequest(
       query = userQuestion,
       k = 3  // Optional: number of context chunks
   )
   val response = apiService.augmentQuery(request)
   ```

3. **Extract Suggested Prompt**: Get the `suggested_prompt` field from the response
   ```kotlin
   val augmentedPrompt = response.suggested_prompt
   ```

4. **Send to LLM**: Instead of sending the raw user input to OpenRouter/your LLM service, send the `suggested_prompt` from the RAG API response
   ```kotlin
   // Instead of: sendToLLM(userQuestion)
   // Do this: sendToLLM(response.suggested_prompt)
   ```

5. **Display Response**: Show the LLM's response to the user

### Why This Approach?

- **Context-Aware**: The RAG API retrieves relevant context from your PDF documents
- **Better Answers**: The LLM receives both the user's question AND relevant context
- **Separation of Concerns**: Your Android app doesn't need to handle vector search logic

### Example Flow Diagram

```
User Question
    ↓
Android App → POST /api/v1/augment
    ↓
RAG Backend (searches vector store)
    ↓
Returns: { original_query, context_chunks, suggested_prompt }
    ↓
Android App extracts suggested_prompt
    ↓
Android App → OpenRouter/LLM API (with suggested_prompt)
    ↓
LLM Response (context-aware answer)
    ↓
Display to User
```

## Code Examples

### Kotlin (Retrofit)

```kotlin
// Data classes
data class AugmentRequest(
    val query: String,
    val k: Int = 3
)

data class AugmentResponse(
    val original_query: String,
    val context_chunks: List<String>,
    val suggested_prompt: String
)

// API Service
interface RAGApiService {
    @POST("/api/v1/augment")
    suspend fun augmentQuery(
        @Body request: AugmentRequest
    ): Response<AugmentResponse>
}

// Usage
val retrofit = Retrofit.Builder()
    .baseUrl("http://10.0.2.2:8000")  // Android emulator
    .addConverterFactory(GsonConverterFactory.create())
    .build()

val apiService = retrofit.create(RAGApiService::class.java)

// In your ViewModel/Repository
suspend fun getAugmentedPrompt(userQuery: String): String {
    val request = AugmentRequest(query = userQuery, k = 3)
    val response = apiService.augmentQuery(request)
    
    if (response.isSuccessful) {
        return response.body()?.suggested_prompt ?: userQuery
    } else {
        // Handle error, fallback to original query
        return userQuery
    }
}
```

### Java (OkHttp)

```java
// Request
String json = "{\"query\":\"" + userQuestion + "\",\"k\":3}";
RequestBody body = RequestBody.create(json, MediaType.parse("application/json"));

Request request = new Request.Builder()
    .url("http://10.0.2.2:8000/api/v1/augment")
    .post(body)
    .build();

// Response handling
try (Response response = client.newCall(request).execute()) {
    if (response.isSuccessful()) {
        String responseBody = response.body().string();
        // Parse JSON and extract suggested_prompt
    }
}
```

## Error Handling

### Recommended Error Handling Strategy

1. **Network Errors**: Handle connection timeouts, network unavailable
2. **503 Errors**: Vector store not loaded - show user-friendly message
3. **500 Errors**: Server errors - log and fallback to direct LLM call
4. **Fallback**: If RAG API fails, send original query directly to LLM

```kotlin
suspend fun getAugmentedPrompt(userQuery: String): String {
    return try {
        val response = apiService.augmentQuery(AugmentRequest(userQuery))
        response.suggested_prompt
    } catch (e: Exception) {
        Log.e("RAG", "Failed to augment query", e)
        // Fallback to original query
        userQuery
    }
}
```

## Testing

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Augment query
curl -X POST http://localhost:8000/api/v1/augment \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?", "k": 3}'
```

### Using Postman

1. Create a new POST request
2. URL: `http://localhost:8000/api/v1/augment`
3. Headers: `Content-Type: application/json`
4. Body (raw JSON):
```json
{
  "query": "Your question here",
  "k": 3
}
```

## Notes for Android Development

1. **Network Security Config**: For Android 9+, you may need to allow cleartext traffic for localhost testing:
   ```xml
   <!-- res/xml/network_security_config.xml -->
   <network-security-config>
       <domain-config cleartextTrafficPermitted="true">
           <domain includeSubdomains="true">10.0.2.2</domain>
       </domain-config>
   </network-security-config>
   ```

2. **Internet Permission**: Ensure you have internet permission in `AndroidManifest.xml`:
   ```xml
   <uses-permission android:name="android.permission.INTERNET" />
   ```

3. **Async Operations**: Always perform API calls on background threads (use coroutines, RxJava, or AsyncTask)

4. **Timeout Configuration**: Set appropriate timeouts for your HTTP client (recommended: 30 seconds)

## Server Status

Before making requests, check the server status:
- **GET /health**: Returns server health and vector store status
- **GET /**: Returns basic server information

## Support

For issues or questions:
1. Check server logs for detailed error messages
2. Verify vector store exists at `./vectorstore` directory
3. Ensure Ollama is running with `nomic-embed-text` model available

