---
layout: post
title: Code Conventions and Personal Hints
excerpt: My Perspective on Multiple Examples of Coding
first_p: |-
    Conventions are created by people. They encapsulate, inherently, these same people's preconceptions and opinions. Bottom line is: they are not necessary logical (although, in many cases, much study is done before adopting a convention). Therefore, it is okay to disagree with conventions and what they state.
toc: true
date: 2017-10-2 00:05:00
lead_image: /assets/images/posts/pep8.webp
tags:
  - Patterns
---

<span>No</span>, this isn't a post saying that you shouldn't create variables called `aux1`, `aux2` and `aux3` (you shouldn't, though).

[Conventions](https://en.wikipedia.org/wiki/Coding_conventions) are created by people. They encapsulate, inherently, these same people's preconceptions and opinions. Bottom line is: they are not necessary logical (although, in many cases, much study is done before adopting a convention). Therefore, it is okay to disagree with conventions and what they state.

That doesn't mean they were created to be broken. They represent an important feature that allows people that don't even speak the same language to create, understand, modified and improved by each other's code.

<figure>
  <img src="/assets/images/posts/pep8.webp" alt="A fun drawing of PEP8 compliant code." class="figure-img img-fluid rounded" />
    <figcaption>A fun drawing of PEP8 complaint code. Available at:
      <a href="https://realpython.com/python-pep8/">realpython/python-pep8</a></figcaption>
</figure>

So what do we do? I've come to the conclusion that it's very important to follow SOME convention. Ideally, the conventions that most people follow in your current programming language. However, you should never stop looking up for new styles that might interest you and why not.

## Some useful references
The most popular conventions are very easily found on the internet. You can get many hints of good programming standards from these files:

* [Java Code Conventions, 1997](http://www.oracle.com/technetwork/java/codeconventions-150003.pdf)
* [c# Coding Conventions](https://msdn.microsoft.com/en-us/library/ff926074.aspx)
* [Python's pep 0008, 2011](https://www.python.org/dev/peps/pep-0008/)

## Doing your part
Conventions aren't everything. It's possible to write unreadable code that follows conventions. Bellow is a list of a few elements that can improve considerably one's code (in my opinion, that is; feel free to disagree).

### Use simple and pleasant names
If your scope is a method with 30 lines, there's no need for huge variable names. Stick with the simple:
```py
# Wrong!
active_users_in_admin_role = User.objects(roles='admin', active=True)
ids_of_active_users_in_admin_role = [user.id for user in active_users_in_admin_role]
repositories_of_active_admin_users = Repository.objects(author__in=ids_of_active_users_in_admin_role)

# Right.
users = User.objects(roles='admin', active=True)
ids_of_users = [u.id for u in users]
repositories = Repository.objects(author__in=ids_of_users)
```

### Be consistent
Don't keep jumping between styles:
```java
boolean canDrive;

if (age >= 16) {
    canDrive = True;
} else {
    canDrive = False;
}

boolean canDrink = age >= 21;
if (canDrive && canDrink)
{
    return true;
}
else
{
    return false;
}
```

Choose one and stick with it:
```java
boolean canDrive = age >= 16;
boolean canDrink = age >= 21;

return canDrive && canDrink;
```

### Give similar names to what's similar and distant names to what's different.
It's a good practice to aggregate similar components and separate different ones. An interesting instance of this is the [Onion Architecture](https://www.develop.com/onionarchitecture).

If you're overriding a method multiple times, try to reuse the code that you've already created, maintaining a logical order:
```csharp
public MusicStreamProvider(Music music) : this(music, DEFAULT_BUFFER_SIZE) { }

public MusicStreamProvider(Music music, int bufferSize = DEFAULT_BUFFER_SIZE)
: this(music.Id, music.Name, music.ChannelId, bufferSize) { }

public MusicStreamProvider(int musicId, string musicName, int channelId, int bufferSize = DEFAULT_BUFFER_SIZE)
{
    LocalFileName = ContextualizedPath(
        DEFAULT_DIRECTORY,
        channelId.ToString(),
        Path.ChangeExtension(
            musicId.ToString(),
            Path.GetExtension(musicName)));
    BufferSize = bufferSize;
}
```

If two methods don't have a similar behavior, they most likely shouldn't have similar names:
```py
#Wrong!
def read_music(music, file):
...

def read_music_s(music, string):
...

#Right.
def read_music_from_file(music):
...
def read_music_from_string(music):
...
```

### Optimize comments
Don't comment code that could be self explanatory:
```csharp
# Wrong!
int a = 10; # skip @a books.
int b = 50; # take @a books.
var page = books.paginate(a, b);

# Right.
int skip = 10;
int take = 50;
var page = books.paginate(skip, take);
```

Be direct, making them as short as possible:
```py
if not_found_in_cache:
# What hasn't been found in cache must be queried from the database.
results_from_database = self.model.objects(id__in=not_found_in_cache)
```

### Consider chaining methods instead of writing multiple statements
As an example of method chaining, [WordsInABook](https://github.com/devraccoon/dojo/blob/master/2.WordsInABook/c%23/WordsInABook/Interpreter.cs#L24)! I'm interested in find out which are the most frequent words in a book:
```csharp
public virtual ICollection MostFrequentWords(int count)
{
    return Book
        .Split()
        .GroupBy(word => word)
        .OrderByDescending(group => group.Count())
        .Take(count)
        .Select(group => group.Key)
        .ToList();
}
```
Of course, this is not the most efficient way to deal with this problem, but in terms of simplicity, I'd say it's good, right? The whole problem was solved in "one" line! In my opinion, *chaining* methods with simple and meaningful names produce a pleasant effect on reading.

[Another solution](https://github.com/devraccoon/dojo/blob/master/2.WordsInABook/ruby/interpreter.rb), by [@cenouro](https://github.com/cenouro):
```ruby
def self.most_frequent_words(str, top_n = 10)
    str.split.sort { |a,b| str.count(b) <=> str.count(a) }.uniq.take(top_n)
end
```

### Don't repeat words.
I really hate verbosity, so I want to take some time on this topic...

Suppose that you're reading a book. The authors can be extremely vague, producing really confusing material, or they could be awfully thoroughly, boring you to death. There is a tiny space between these two extremes that will create an enjoyable story.

Code is writing. As the author, you must be able to get to that enjoyable region. Achieving that will make your code less stressful and more pleasant to read, hence increasing quality.

We tend to repeat ourselves very often.
```csharp
UsersRepository usersRepository = new UsersRepository(Context);
usersRepository.AddUser(user);
```
After four times, I THINK I GOT that this is a repository of users!

- Why repeat the name so many times?
- Why do we have to call it `usersRepository`? Doesn't the plural form `users` already indicates a collection to us? Besides, we'll have to write down a huge name every time.
- Why is there a method called `.AddUser()`? This is a repository of users! The parameter of the method is `User user`! Did someone think we would implement the method `.AddHorse(Horse horse)` someday?

What if it were like this:
```csharp
var users = new UsersRepository(Context);
users.Add(user);
```

> **#1** Never go full retard and implement a method called `.AddHorse(Horse horse)` on a class named `UsersRepository`.

Another example:
```csharp
public class EnrollmentService
{
    public async Task OfStudent(string id)
    {
        return await Db.Enrollments
            .Where(e => e.StudentId == id)
            .ToListAsync();
    }
}
```

Do you think `EnrollmentsService.OfStudent()` is a confusing name? Look at the **signature** of the method, not only the name. Clearly, it returns a collection of `Enrollments`. That can only mean that I'm returning a list of enrollments of a given student `id`.
This is how I'd use it:
```csharp
var student = ...
var enrollments = EnrollmentService();
return await enrollments.OfStudent(student.Id);
```
Look at the third line. That's almost natural! Are we seriously still considering that `await enrollmentsRepository.GetEnrollmentsOfStudent(student.Id)` might be better?

## Switch languages often, until you find one that you like.
### Java
Sometimes, your ideas don't really match a programming language standards. Personally, I'm not a big fan of Java. Consider the example:
```java
@Document
public class Course {
    private ObjectId id;
    private String name;
    private Professor professor;

    public getId() { return this.id; }
    public setId(ObjectId id) { this.id = id; }
    public getName() { return this.name; }
    public setName(String name) { this.name = name; }
    public getName() { return this.name; }
    public getProfessor() { return this.professor; }
    public setProfessor(Professor professor) { this.id = id; }
}

String name = ...
Professor professor = ...

Course course = new Course();
course.setName(name);
course.setProfessor(professor);
```

How many times did we repeat the word `course` on the initializing? And why do I have to say `.setProfessor`, given that the name of the parameter in the method's signature is already `professor`? Isn't obvious that I'm setting a `professor`?

> **#2** If Java were a sentence, it would be *"Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo"*. It's grammatically correct, but it's fucking dumb. It assumes that all programmers are retarded people that can't read, and so it must repeat the same words over and over. This is called [verbosity](http://programmers.stackexchange.com/questions/141175/why-is-verbosity-bad-for-a-programming-language).

### C# slightly different approach
C# is similar to Java in many ways. Repetition is not one of them:
```csharp
public class Course
{
    public int Id { get; set; }
    public string Name { get; set; }
    public virtual Professor Professor { get; set; }
}

name = ...
professor = ...

var course = new Course
{
    Name = name,
    Professor = professor
};
```

Nice, right? We've eliminated one usage of the word course with `var` and the sequential calls for methods `set` because of the way C# allows us to instantiate classes.

C# also allows us to substitute the `get` and `set` keywords by `{}` and write any pre-assingment validation that we might eventually need. In other words, they work as [mutator methods](https://en.wikipedia.org/wiki/Mutator_method).

### The Pythonic way.
In Python, nothing is private, only members declared with an underscore, which means "be careful, use it only if you deeply understands the class functioning". It's a little bit more rational, as it grants full modification freedom to future programmers, if they eventually come up with a better way to do things.
```python
class Course(mongoengine.Document):
name = mongoengine.StringField()
professor = mongoengine.EmbeddedDocument(Professor)

name = ...
professor = ...

course = Course(name=name, professor=professor)
```

## Conclusion
This post was very biased, I know. Let's be honest, though: we all have our preferences when it comes to languages, styles, environments etc. I believe that programming freedom yields better results than restriction and "protection", and that's why Java isn't the best language for me. In the other hand, I'm pretty positive there is a huge community that have very good reasons to like Java.

We're all different. "Our world isn't perfect, and that's what makes it so beautiful" - [Mustang](http://fma.wikia.com/wiki/Roy_Mustang).

> **#3** Finally, I'll leave this reference: [PEP 20 -- The Zen of Python](https://www.python.org/dev/peps/pep-0020/). You might find it very reasonable. :-)
