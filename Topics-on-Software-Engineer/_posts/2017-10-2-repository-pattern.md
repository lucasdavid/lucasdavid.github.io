---
layout: post
title: Repository Pattern
excerpt: Understanding A Data Abstraction Pattern
date: 2017-10-2 00:07:00
color: indigo darken-4
---

## Introduction
Real life applications might have to communicate with more than one data source. Data might be coming from/going to different places and it's hard to keep it simple, as you have to implement the interaction with all of these places individually. The [repository pattern](http://docs.spring.io/spring-data/data-commons/docs/1.6.1.RELEASE/reference/html/repositories.html) abstracts the data layer (how you store and retrieve your resources), making it transparent to the business layer. It's important to keep in mind that it does not replace an [ORM](https://ycoding.wordpress.com/2015/05/03/object-relational-mapping/), which is specifically defined to map model classes to database entities.

## Example #1: invisible database
[Spring](https://spring.io/) has one of the greatest Repository Pattern implementations, although these are specifically designed to abstract databases. Here, the repositories are interfaces and the queries are built based on the methods signatures.
```java
public interface ProductRepository extends CrudRepository<Product, Long>, QueryDslPredicateExecutor<Product> {
    // Returns a {@link Page} of {@link Product}s having a description which contains the given snippet.
    Page<Product> findByDescriptionContaining(String description, Pageable pageable);


    // Returns all {@link Product}s having the given attribute.
    @Query("{ ?0 : ?1 }")
    List<Product> findByAttributes(String key, String value);
}
```
*Example taken from [spring-data-book](https://github.com/spring-projects/spring-data-book/blob/master/mongodb/src/main/java/com/oreilly/springdata/mongodb/core/ProductRepository.java).*

Spring understands what we're trying to achieve by the methods signature. For instance, when we execute the method `.findByDescriptionContaining()`, it's clear that we want to "retrieve products with descriptions that contain the substring `description`".

Checkout more examples in these repositories: [spring-data-book](https://github.com/spring-projects/spring-data-book) and [spring-jpa-examples](https://github.com/spring-projects/spring-data-jpa-examples).

## Example 2: Performance boost
Architecturally, the Repository Pattern is higher than an ORM, as it abstracts any data source: databases, APIs, user's input etc. There's nothing that prevents repositories from making usage of an ORM, although not always that's the best way to go. Remember when I said that ORMs are recommended in the great majority of cases? Let's check out an counter-example. Imagine that we have a collection of products and we want to insert them on the database, costing $1.
```py
products = Product.objects.all()

for product in products:
    product.price = 1
    product.save()
```

Notice that this code will make **N** queries to the database, where **N** is the length of the `products` list. That's not great, considering a single raw SQL query would perform much better (it doesn't require application processing and it only "round-trips" once).
```py
session.query("""
    insert into products values (...);
    insert into products values (...);
    insert into products values (...);
    ...
""")
```

In general, aggregating SQL queries and sending them all together performs better than sending each one sequentially. That becomes evident when we have collections, as there are many queries to be sent. We can't improve this using an ORM because of its nature: it is a mapping between an object and a relation entity. But what about a repository? Many repository implementations give us methods such as `.create_many` and `.update_many`, allowing us to handle collections efficiently.
```csharp
await products.ForEachAsync(product => product.Price = 1);

Db.Products.AddRange(products);

await Db.SaveChangesAsync();
```

In practical terms, [Entity Framework](www.asp.net/entity-framework) already implements the Repository Pattern with the `DbSet` class and the [Unit Of Work Pattern](https://msdn.microsoft.com/en-us/magazine/dd882510.aspx) with the `DbContext` class. When it comes to performance, Entity Framework can do an impressive job! All the inserted products are sent to the database at once, upon `.SaveChangesAsync()` call.

## Example #3: multiple data sources
A final example. Imagine that we own an e-commerce and we want to create an application that can retrieve products on it and store in a local database. We'll have to handle two very different data layers: our own application's API, from which we'll retrieve the products, and our own database.

For simplicity, assume our application's API is a [REST API](http://en.wikipedia.org/wiki/Representational_state_transfer) and it doesn't require any form of authentication. Being an REST API implies that we can access products by making an HTTP request to an URL like **http: //api.our-very-e-commerce.com/products/{id}** with the `GET` method. We can also create or update products with the methods `POST` and `PUT`, respectively. Finally, we can delete a product with the method `DELETE`.

### A first attempt
```py
def ad_creation(product_id):
    url = 'http://api.our-very-own-e-commerce.com/products/'

    response = requests.get(url)
    assert response

    product_data_list = response.json()

    for data in product_data_list:
        Product(**data).save()
```

Without the Repository Pattern, our business logic depends heavily on the implementation of data sources. With these components hardly coupled, changes in the data persistence logic will greatly affect our entire application. Let's fix it.

### Now using RP
First, we'll abstract all of this by using a base `Repository` class:
```py
class Repository(object):
    __class__ = abc.ABCMeta

    def all():
        raise NotImplemented

    def find(self, id):
        raise NotImplemented

    def create(self, entity):
        raise NotImplemented

    def update(self, entiy):
        raise NotImplemented

    def delete(self, id):
        raise NotImplemented
```

And why did we do this? Simple: we can now inject a object of a class that inherits from `Repository` inside our code and simply call these methods that we've just defined. That's called **inclusion polymorphism** and it helps us to write generic code. Please take your time to check this [Polymorphism article at wikipedia](http://en.wikipedia.org/wiki/Polymorphism_%28computer_science%29).

Let's implement our first inherited class. The one that will communicate with our e-commerce API.

```py
class APIRepository(Repository):
    def __init__(self, base_api_url, resource):
        self.url = '/'.join(base_api_url, resource)

    def _validate(self, response):
        if not response:
            # If the response.status_code indicates an error, displays it.
            url = response.url
            status_code = str(response.status_code)
            try:
                json = str(response.json())
            except ValueError:
                json = response.content

            raise RuntimeError('Error when requesting ' + url + ':'
                               + 'nStatus-code: ' + status_code
                               + 'nDetails: ' + json)

        return self

    @staticmethod
    def _unwrap(response):
        if response.status_code == requests.codes.no_content:
            return {}

        return response.json()

    def _result(self, response):
        return self
            ._validate(response)
            ._unwrap(response)

    def all(self):
        response = requests.get(self.url)

        return self._result(response)

    def find(self, id):
        response = requests.get(
            '/'.join(
                self.url,
                str(id)))

        return self._result(response)

    def create(self, entity):
        response = requests.post(self.url, data=entity)

        return self._result(response)

    def update(self, entity):
        response = requests.get(
            '/'.join(
                self.url,
                str(id)),
            data=entity)

        return self._result(response)

    def delete(self, id):
        response = requests.get(
            '/'.join(
                self.url,
                str(id)))

        return self._result(response)
```

`APIRepository` inherits `Repository` and overrides its abstract methods. All overriding methods simply make an HTTP request using the [requests](http://docs.python-requests.org/en/latest/) lib. For instance, in `.update(self, entity)`, we are:

* Merging the URL with the expected product id, yielding **http: //.../products/48**, for example.
* Making a request to that merged URL, passing into the request's body all the data that will be used to update the product #48.
* Checking if the response contains any errors and, finally, unwrapping its response.

We can now use it like this:
```py
products = APIRepository('products')
product_48 = products.find(48)
```

Simple, right? Notice that we've created an interface that works not only with products, but any resource that our API provides. If, in the future, we want to retrieve another resources, such as users, we can do it by simply instantiating `APIRepository('users')` and calling the method `.find()`.

> **#1** Always separate responsibilities and make your components [simple, stupid](http://en.wikipedia.org/wiki/KISS_principle). When they are really small, you can easily identify similarities and code replication. That's when you can generalize and keep the number of lines of code to a minimum. [Don't repeat yourself](http://en.wikipedia.org/wiki/Don%27t_repeat_yourself).

We still have to implement the repository that'll persist the retrieved products to our local database. Let's assume that we're working with [MongoDB](https://www.mongodb.org/) with the [MongoEngine](http://mongoengine.org/) ORM.
```py
class MongoRepository(Repository):
    def __init__(self, model)
        self.model = model

    def all():
        return model.objects

    def find(self, id):
        return model.objects(id=id).first()

    def create(self, entity):
        return model(**entity).save()

    def update(self, entiy):
        return model
            .objects(id=entity.id)
            .first()
            .update(**entity)
            .save()

    def delete(self, id):
        return model
            .objects(id=id)
            .first()
            .delete()
```
Notice how our `MongoRepository` also overrides the expected methods! However, this time, it will operate over an MongoDB document.

We can finally implement our logic as this:

```py
def copy_products(in_products, out_products):
    products = in_products.all()

    for product in products:
        out_products.add(products)

in = APIRepository('products')
out = MongoRepository(Product)

copy_products(in, out)
```

Neat! We stopped worrying on how the data is handled and focused entirely on our problem, which was to copy the products. Additionally, our code is now so flexible! It can handle business logic changes with almost zero difficult! Given the products back to the API would be so simple as:
```py
in = MongoRepository('products')
out = APIRepository(Product)

copy_products(in, out)
```

## Conclusion
I hope the examples above have shown you some advantages of using the Repository Pattern. If I were to summarize, I'd say that with the *RP*, you can:

 * Focus on the business logic, making everything much more clear.
 * Replace highly-coupled components by unit-testable, [nice and soft](https://www.youtube.com/watch?v=XifUlSyQ2CI) dependencies.
 * Generalize your code, abbreviating it and preventing you from from repeating yourself and creating huge methods, making everyone - including yourself - very sad.

That's it! If you want to exercise yourself, why don't you try to implement the methods `.save_many()` and `.update_many()` in the `MongoRepository`? :-)

P.s.: please be aware that these implementations, although strongly inspired in real-life cases, were simplified and are now merely toy examples. Please research the communities guidelines before making any implementations of your own.
