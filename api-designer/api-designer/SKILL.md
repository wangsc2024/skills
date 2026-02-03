---
name: api-designer
description: |
  Design RESTful and GraphQL APIs with proper structure, naming conventions, and documentation. Creates OpenAPI/Swagger specs, defines endpoints, and plans API versioning.
  Use when: designing APIs, creating API specifications, planning backend endpoints, documenting API contracts, or when user mentions API, REST, GraphQL, endpoint, OpenAPI, Swagger, API設計, 接口.
  Triggers: "API", "REST", "endpoint", "GraphQL", "OpenAPI", "Swagger", "設計API", "接口設計", "後端接口"
---

# API Designer

Design production-ready APIs following industry best practices and standards.

## Design Process

1. **Identify Resources** - What entities does the API manage?
2. **Define Operations** - What actions can be performed?
3. **Design Endpoints** - How are resources accessed?
4. **Plan Responses** - What data is returned?
5. **Handle Errors** - How are failures communicated?
6. **Document** - How do developers understand the API?

## RESTful API Design

### Resource Naming

```
# Good: Nouns, plural, lowercase, kebab-case
GET    /users
GET    /users/{id}
GET    /users/{id}/orders
GET    /order-items

# Bad: Verbs, singular, camelCase
GET    /getUser
GET    /user/{id}
GET    /getUserOrders
GET    /orderItems
```

### HTTP Methods

| Method | Purpose | Idempotent | Request Body |
|--------|---------|------------|--------------|
| GET | Retrieve resource(s) | Yes | No |
| POST | Create resource | No | Yes |
| PUT | Replace resource | Yes | Yes |
| PATCH | Partial update | Yes | Yes |
| DELETE | Remove resource | Yes | No |

### Endpoint Patterns

```yaml
# Collection operations
GET    /users              # List all users
POST   /users              # Create user

# Single resource operations
GET    /users/{id}         # Get user by ID
PUT    /users/{id}         # Replace user
PATCH  /users/{id}         # Update user fields
DELETE /users/{id}         # Delete user

# Nested resources
GET    /users/{id}/orders  # User's orders
POST   /users/{id}/orders  # Create order for user

# Actions (when REST doesn't fit)
POST   /users/{id}/activate
POST   /orders/{id}/cancel
```

### Query Parameters

```yaml
# Pagination
GET /users?page=2&limit=20
GET /users?offset=40&limit=20
GET /users?cursor=abc123&limit=20

# Filtering
GET /users?status=active
GET /users?role=admin&created_after=2024-01-01

# Sorting
GET /users?sort=created_at:desc
GET /users?sort=-created_at,name

# Field selection
GET /users?fields=id,name,email

# Search
GET /users?q=john
GET /users?search=john@example
```

### Response Structure

```json
// Single resource
{
  "data": {
    "id": "usr_123",
    "type": "user",
    "attributes": {
      "name": "John Doe",
      "email": "john@example.com",
      "created_at": "2024-01-15T10:30:00Z"
    }
  }
}

// Collection
{
  "data": [...],
  "meta": {
    "total": 150,
    "page": 2,
    "limit": 20,
    "total_pages": 8
  },
  "links": {
    "self": "/users?page=2",
    "first": "/users?page=1",
    "prev": "/users?page=1",
    "next": "/users?page=3",
    "last": "/users?page=8"
  }
}
```

### Error Responses

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": [
      {
        "field": "email",
        "message": "Invalid email format",
        "code": "INVALID_FORMAT"
      },
      {
        "field": "password",
        "message": "Must be at least 8 characters",
        "code": "MIN_LENGTH"
      }
    ]
  },
  "request_id": "req_abc123"
}
```

### HTTP Status Codes

```yaml
# Success
200 OK              # Successful GET, PUT, PATCH
201 Created         # Successful POST (include Location header)
204 No Content      # Successful DELETE

# Client Errors
400 Bad Request     # Validation error
401 Unauthorized    # Missing/invalid authentication
403 Forbidden       # Authenticated but not authorized
404 Not Found       # Resource doesn't exist
409 Conflict        # Resource conflict (duplicate)
422 Unprocessable   # Semantic validation error
429 Too Many Requests # Rate limited

# Server Errors
500 Internal Error  # Unexpected server error
502 Bad Gateway     # Upstream service error
503 Unavailable     # Service temporarily down
```

## OpenAPI Specification

```yaml
openapi: 3.0.3
info:
  title: User Management API
  version: 1.0.0
  description: API for managing users and their resources

servers:
  - url: https://api.example.com/v1
    description: Production
  - url: https://api.staging.example.com/v1
    description: Staging

paths:
  /users:
    get:
      summary: List users
      operationId: listUsers
      tags: [Users]
      parameters:
        - name: page
          in: query
          schema:
            type: integer
            default: 1
        - name: limit
          in: query
          schema:
            type: integer
            default: 20
            maximum: 100
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UserList'

    post:
      summary: Create user
      operationId: createUser
      tags: [Users]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateUserRequest'
      responses:
        '201':
          description: User created
          headers:
            Location:
              schema:
                type: string
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
        '400':
          $ref: '#/components/responses/ValidationError'

components:
  schemas:
    User:
      type: object
      required: [id, email, name]
      properties:
        id:
          type: string
          example: usr_123
        email:
          type: string
          format: email
        name:
          type: string
        created_at:
          type: string
          format: date-time

    CreateUserRequest:
      type: object
      required: [email, name, password]
      properties:
        email:
          type: string
          format: email
        name:
          type: string
          minLength: 1
          maxLength: 100
        password:
          type: string
          minLength: 8

  responses:
    ValidationError:
      description: Validation error
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'

  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT

security:
  - bearerAuth: []
```

## API Versioning

```yaml
# URL path versioning (recommended)
GET /v1/users
GET /v2/users

# Header versioning
GET /users
Accept: application/vnd.api+json; version=2

# Query parameter
GET /users?version=2
```

## Rate Limiting Headers

```yaml
X-RateLimit-Limit: 1000        # Requests per window
X-RateLimit-Remaining: 998     # Remaining requests
X-RateLimit-Reset: 1640000000  # Unix timestamp of reset
Retry-After: 60                # Seconds to wait (on 429)
```

## Authentication Patterns

```yaml
# API Key (header)
X-API-Key: sk_live_abc123

# Bearer Token (JWT)
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...

# OAuth 2.0 flows
- Authorization Code (web apps)
- Client Credentials (machine-to-machine)
- PKCE (mobile/SPA)
```

## Checklist

- [ ] Resources use plural nouns
- [ ] Endpoints are hierarchical and logical
- [ ] HTTP methods match operations
- [ ] Consistent response structure
- [ ] Proper status codes used
- [ ] Errors include actionable details
- [ ] Pagination for collections
- [ ] Filtering and sorting supported
- [ ] Versioning strategy defined
- [ ] Authentication documented
- [ ] Rate limiting headers included
- [ ] OpenAPI spec complete
